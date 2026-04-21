"""
BrainDB ↔ TradingAgents Bridge

Connects FinMEM's layered memory system with TradingAgents' multi-agent
decision pipeline:

1. Before TradingAgents runs: query BrainDB for relevant memories and
   inject them into the analyst prompts via the initial state.

2. After TradingAgents runs: feed the analyst reports back into BrainDB
   as new short-term memories for future cycles.
"""

import logging
from datetime import date, datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def query_brain_context(
    brain,
    ticker: str,
    top_k: int = 5,
) -> Dict[str, str]:
    """Query all BrainDB layers and return formatted context blocks.

    Args:
        brain: BrainDB instance.
        ticker: Stock ticker.
        top_k: Memories to retrieve per layer.

    Returns:
        Dict with keys: short, mid, long, reflection — each a formatted string.
    """
    context = {}
    query = f"trading analysis {ticker} market conditions price movement news"

    for layer_name, query_fn in [
        ("short", brain.query_short),
        ("mid", brain.query_mid),
        ("long", brain.query_long),
        ("reflection", brain.query_reflection),
    ]:
        try:
            texts, ids = query_fn(query, top_k, ticker)
            if texts:
                context[layer_name] = "\n".join(
                    f"  [{layer_name.upper()} #{mid}] {t[:400]}"
                    for t, mid in zip(texts, ids)
                )
            else:
                context[layer_name] = f"(No {layer_name}-term memories for {ticker})"
        except Exception as e:
            logger.warning(f"[Bridge] Failed to query {layer_name} for {ticker}: {e}")
            context[layer_name] = f"(Query failed: {e})"

    return context


def format_brain_context_for_analysts(context: Dict[str, str]) -> str:
    """Format BrainDB context into a single text block for injection into prompts.

    This gets prepended to the TradingAgents initial human message so all
    analysts see the memory context alongside the ticker name.
    """
    sections = []
    sections.append("=== FinMEM Memory Context (from layered memory system) ===")

    if context.get("short"):
        sections.append(f"\n--- Short-Term Memories (recent news, daily) ---\n{context['short']}")
    if context.get("mid"):
        sections.append(f"\n--- Mid-Term Memories (quarterly trends) ---\n{context['mid']}")
    if context.get("long"):
        sections.append(f"\n--- Long-Term Memories (annual fundamentals) ---\n{context['long']}")
    if context.get("reflection"):
        sections.append(f"\n--- Reflection Memories (past decision lessons) ---\n{context['reflection']}")

    sections.append("\n=== End Memory Context ===\n")
    return "\n".join(sections)


def create_enriched_initial_state(
    ticker: str,
    trade_date: str,
    brain=None,
    top_k: int = 5,
    regime: str = "SIDEWAYS",
) -> Dict[str, Any]:
    """Create TradingAgents initial state enriched with BrainDB memories.

    The key insight: TradingAgents' initial state starts with a human message
    containing just the company name. We augment this with memory context
    so every analyst in the pipeline sees FinMEM's accumulated knowledge.
    """
    from tradingagents.agents.utils.agent_states import (
        InvestDebateState,
        RiskDebateState,
    )

    memory_block = ""
    if brain is not None:
        context = query_brain_context(brain, ticker, top_k)
        memory_block = format_brain_context_for_analysts(context)

    enriched_message = (
        f"{ticker}\n\n"
        f"Market Regime: {regime}\n"
        f"Trade Date: {trade_date}\n\n"
        f"{memory_block}"
    )

    return {
        "messages": [("human", enriched_message)],
        "company_of_interest": ticker,
        "trade_date": str(trade_date),
        "investment_debate_state": InvestDebateState(
            {
                "bull_history": "",
                "bear_history": "",
                "history": "",
                "current_response": "",
                "judge_decision": "",
                "count": 0,
            }
        ),
        "risk_debate_state": RiskDebateState(
            {
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "history": "",
                "latest_speaker": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 0,
            }
        ),
        "market_report": "",
        "fundamentals_report": "",
        "sentiment_report": "",
        "news_report": "",
    }


def ingest_ta_reports_into_brain(
    brain,
    ticker: str,
    trade_date,
    final_state: Dict[str, Any],
) -> List[int]:
    """Feed TradingAgents analyst reports back into BrainDB as short-term memories.

    Each analyst report becomes a new memory in the short-term layer,
    so future cycles benefit from accumulated multi-agent analysis.

    Returns list of new memory IDs.
    """
    if brain is None:
        return []

    if isinstance(trade_date, str):
        trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()

    new_ids = []
    report_fields = [
        ("market_report", "Market Analysis"),
        ("sentiment_report", "Sentiment Analysis"),
        ("news_report", "News Analysis"),
        ("fundamentals_report", "Fundamentals Analysis"),
    ]

    for field, label in report_fields:
        report_text = final_state.get(field, "")
        if report_text and len(report_text.strip()) > 50:
            summary = f"[{label} — {ticker} — {trade_date}] {report_text[:1500]}"
            try:
                ids = brain.add_memory_short(ticker, trade_date, summary)
                new_ids.extend(ids)
                logger.info(f"[Bridge] Ingested {label} for {ticker} → memory ID(s) {ids}")
            except Exception as e:
                logger.warning(f"[Bridge] Failed to ingest {label}: {e}")

    investment_plan = final_state.get("investment_plan", "")
    if investment_plan and len(investment_plan.strip()) > 50:
        try:
            ids = brain.add_memory_short(
                ticker, trade_date,
                f"[Investment Plan — {ticker} — {trade_date}] {investment_plan[:1500]}"
            )
            new_ids.extend(ids)
        except Exception:
            pass

    final_decision = final_state.get("final_trade_decision", "")
    if final_decision and len(final_decision.strip()) > 20:
        try:
            ids = brain.add_memory_short(
                ticker, trade_date,
                f"[Final Decision — {ticker} — {trade_date}] {final_decision[:1500]}"
            )
            new_ids.extend(ids)
        except Exception:
            pass

    return new_ids
