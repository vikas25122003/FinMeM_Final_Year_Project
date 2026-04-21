"""
Reflection Bridge — Connects both reflection systems after trade outcomes.

After each trading cycle, feeds P&L results into:
1. TradingAgents' BM25 memory (reflect_and_remember)
2. FinMEM BrainDB access counters (+1/-1 feedback)
3. FinMEM reflection memory (stores lessons learned)
4. Obj2 reflection logger (JSONL for importance retraining)
"""

import logging
from datetime import date, datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def compute_trade_returns(
    positions_before: Dict[str, Dict[str, Any]],
    positions_after: Dict[str, Dict[str, Any]],
    ticker: str,
) -> float:
    """Compute percentage return for a ticker between two position snapshots."""
    before = positions_before.get(ticker, {})
    after = positions_after.get(ticker, {})

    before_value = before.get("market_value", 0)
    after_value = after.get("market_value", 0)

    if before_value > 0:
        return (after_value - before_value) / before_value
    elif after_value > 0:
        entry = after.get("avg_entry", 0)
        current = after.get("current_price", 0)
        if entry > 0:
            return (current - entry) / entry
    return 0.0


def reflect_trading_agents(
    ta_graph,
    returns: float,
) -> None:
    """Run TradingAgents' built-in reflection on all role memories.

    This updates BM25 memories for bull, bear, trader, judge, and
    portfolio manager based on trade outcomes.
    """
    try:
        ta_graph.reflect_and_remember(returns)
        logger.info(f"[Reflection] TradingAgents reflection completed (returns={returns:.4f})")
    except Exception as e:
        logger.warning(f"[Reflection] TradingAgents reflection failed: {e}")


def reflect_finmem_brain(
    brain,
    ticker: str,
    trade_date,
    decision: str,
    returns: float,
    memory_ids_used: List[int],
    rationale: str = "",
) -> None:
    """Update FinMEM BrainDB based on trade outcomes.

    1. Updates access counters (boost if profitable, penalize if not)
    2. Stores a reflection memory summarizing the outcome
    3. Logs to Obj2 JSONL for importance model retraining
    """
    if brain is None:
        return

    if isinstance(trade_date, str):
        trade_date_obj = datetime.strptime(trade_date, "%Y-%m-%d").date()
    else:
        trade_date_obj = trade_date

    feedback = 1 if returns > 0 else -1

    for layer in [brain.short_term_memory, brain.mid_term_memory,
                  brain.long_term_memory, brain.reflection_memory]:
        for symbol in layer.universe:
            if symbol != ticker:
                continue
            for record in layer.universe[symbol]["score_memory"]:
                if record["id"] in memory_ids_used:
                    record["access_counter"] += feedback
                    if feedback > 0:
                        record["important_score"] = min(
                            1.0, record["important_score"] * 1.05
                        )
                    else:
                        record["important_score"] = max(
                            0.01, record["important_score"] * 0.95
                        )

    direction_emoji = "+" if returns > 0 else "-"
    reflection_text = (
        f"[Reflection — {ticker} — {trade_date_obj}] "
        f"Decision: {decision} | Return: {returns*100:+.2f}% ({direction_emoji}) | "
        f"Memories used: {len(memory_ids_used)} | "
        f"{rationale[:500]}"
    )
    try:
        brain.add_memory_reflection(ticker, trade_date_obj, reflection_text)
        logger.info(f"[Reflection] BrainDB reflection stored for {ticker}")
    except Exception as e:
        logger.warning(f"[Reflection] BrainDB reflection storage failed: {e}")

    _log_obj2_reflection(
        trade_date=str(trade_date_obj),
        ticker=ticker,
        decision=decision,
        memory_ids_used=memory_ids_used,
        rationale=rationale,
        cumulative_return=returns,
    )


def _log_obj2_reflection(
    trade_date: str,
    ticker: str,
    decision: str,
    memory_ids_used: List[int],
    rationale: str,
    cumulative_return: float,
) -> None:
    """Log to Obj2 JSONL files for importance classifier retraining."""
    try:
        from agentic.obj2_importance.logger import log_reflection
        log_reflection(
            date=trade_date,
            ticker=ticker,
            decision=decision,
            memory_ids_used=memory_ids_used,
            rationale=rationale,
            cumulative_return=cumulative_return,
        )
    except Exception as e:
        logger.warning(f"[Reflection] Obj2 logging failed: {e}")


def run_full_reflection(
    ta_graph,
    brain,
    ticker: str,
    trade_date: str,
    decision: str,
    returns: float,
    memory_ids_used: List[int],
    final_state: Dict[str, Any],
) -> None:
    """Run the complete reflection pipeline for one ticker after trading.

    Orchestrates both TradingAgents and FinMEM reflection systems.
    """
    logger.info(f"[Reflection] Starting full reflection for {ticker} "
                f"(decision={decision}, return={returns*100:+.2f}%)")

    reflect_trading_agents(ta_graph, returns)

    rationale = final_state.get("final_trade_decision", "")[:500]
    reflect_finmem_brain(
        brain=brain,
        ticker=ticker,
        trade_date=trade_date,
        decision=decision,
        returns=returns,
        memory_ids_used=memory_ids_used,
        rationale=rationale,
    )

    logger.info(f"[Reflection] Full reflection completed for {ticker}")
