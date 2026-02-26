"""
Reflection Module — Paper-Faithful Implementation

The core "working memory" mechanism from the FinMEM paper.
Performs three operations:
1. Summarization — condenses news into key insights via LLM
2. Observation — structures price data  
3. Reflection — queries all memory layers, LLM makes buy/hold/sell decision

After reflection, the LLM identifies "pivotal" memory IDs → +5 importance bonus
(paper's Guardrails AI equivalent).
"""

import json
import logging
from datetime import date, datetime
from typing import Optional, Dict, Any, List, Union, Tuple

from ..llm_client import LLMClient
from ..memory.layered_memory import BrainDB
from .prompts import (
    train_investment_info_prefix,
    train_prompt,
    test_investment_info_prefix,
    test_prompt,
    test_sentiment_explanation,
    test_momentum_explanation,
)

logger = logging.getLogger(__name__)


# ── Working Memory Operation 1: Summarization ──

def summarize_news(
    llm: LLMClient,
    news_list: List[str],
    symbol: str,
    cur_date: Union[date, datetime],
) -> str:
    """Summarize news articles into key financial insights via LLM.
    
    Paper: News is not stored raw — it passes through a summarization step
    in the working memory before being committed to the shallow layer.
    
    Args:
        llm: LLM client for summarization.
        news_list: Raw news headlines/articles.
        symbol: Stock ticker.
        cur_date: Current date.
        
    Returns:
        Summarized insight text for memory storage.
    """
    if not news_list:
        return ""
    
    # Format news for summarization
    news_text = "\n".join(f"- {n}" for n in news_list[:10])  # Cap at 10
    
    prompt = (
        f"You are a financial analyst. Summarize these news items for {symbol} "
        f"on {cur_date} into concise, market-relevant insights.\n\n"
        f"News:\n{news_text}\n\n"
        f"Provide a brief summary (2-3 sentences max) focusing on:\n"
        f"1. Overall sentiment (bullish/bearish/neutral)\n"
        f"2. Key events or catalysts\n"
        f"3. Potential market impact\n\n"
        f"Respond with only the summary text, no markup."
    )
    
    try:
        summary = llm.chat(prompt)
        return f"[{cur_date}] News Summary for {symbol}: {summary.strip()}"
    except Exception as e:
        logger.warning(f"News summarization failed: {e}")
        # Fallback: concatenate first few headlines
        combined = "; ".join(n[:100] for n in news_list[:3])
        return f"[{cur_date}] News for {symbol}: {combined}"


# ── Working Memory Operation 2: Observation ──

def observe_price(
    llm: LLMClient,
    symbol: str,
    cur_date: Union[date, datetime],
    cur_price: float,
    price_history: List[float],
    momentum: Optional[int] = None,
) -> str:
    """Structured price observation via LLM.
    
    Paper: Working memory "Observation" operation — analyzes price patterns.
    Creates a structured observation that captures current market conditions.
    
    Args:
        llm: LLM client.
        symbol: Stock ticker.
        cur_date: Current date.
        cur_price: Current close price.
        price_history: Recent price history (last N days).
        momentum: 3-day momentum signal (-1, 0, 1).
        
    Returns:
        Observation text for memory storage.
    """
    # Build price context
    recent = price_history[-5:] if len(price_history) >= 5 else price_history
    price_changes = []
    for i in range(1, len(recent)):
        chg = ((recent[i] - recent[i-1]) / recent[i-1]) * 100
        price_changes.append(f"{chg:+.2f}%")
    
    price_str = ", ".join(f"${p:.2f}" for p in recent)
    changes_str = ", ".join(price_changes) if price_changes else "N/A"
    
    momentum_str = {-1: "NEGATIVE", 0: "FLAT", 1: "POSITIVE"}.get(momentum, "UNKNOWN")
    
    prompt = (
        f"You are a financial analyst observing {symbol} on {cur_date}.\n\n"
        f"Current Price: ${cur_price:.2f}\n"
        f"Recent Prices (last {len(recent)} days): {price_str}\n"
        f"Daily Changes: {changes_str}\n"
        f"3-Day Momentum: {momentum_str}\n\n"
        f"Provide a brief observation (1-2 sentences) on the current price pattern. "
        f"Note any trends, support/resistance levels, or notable movements. "
        f"Respond with only the observation text."
    )
    
    try:
        observation = llm.chat(prompt)
        return (
            f"[{cur_date}] Price Observation for {symbol}: "
            f"Price=${cur_price:.2f}, Momentum={momentum_str}. "
            f"{observation.strip()}"
        )
    except Exception as e:
        logger.warning(f"Price observation failed: {e}")
        return (
            f"[{cur_date}] Price Observation for {symbol}: "
            f"Price=${cur_price:.2f}, "
            f"Recent changes: {changes_str}, "
            f"Momentum: {momentum_str}"
        )


def _format_memories_for_prompt(
    short_memory: List[str],
    short_memory_ids: List[int],
    mid_memory: List[str],
    mid_memory_ids: List[int],
    long_memory: List[str],
    long_memory_ids: List[int],
    reflection_memory: List[str],
    reflection_memory_ids: List[int],
) -> str:
    """Format retrieved memories into a prompt string."""
    parts = []

    if short_memory:
        parts.append("### Short-term Memory (Recent News & Events)")
        for mid, text in zip(short_memory_ids, short_memory):
            parts.append(f"  [{mid}] {text.strip()}")
        parts.append("")

    if mid_memory:
        parts.append("### Mid-term Memory (Quarterly Trends & Filings)")
        for mid, text in zip(mid_memory_ids, mid_memory):
            parts.append(f"  [{mid}] {text.strip()}")
        parts.append("")

    if long_memory:
        parts.append("### Long-term Memory (Fundamentals & Annual Data)")
        for mid, text in zip(long_memory_ids, long_memory):
            parts.append(f"  [{mid}] {text.strip()}")
        parts.append("")

    if reflection_memory:
        parts.append("### Reflection Memory (Past Trading Insights)")
        for mid, text in zip(reflection_memory_ids, reflection_memory):
            parts.append(f"  [{mid}] {text.strip()}")
        parts.append("")

    return "\n".join(parts)


def _add_momentum_info(momentum: Optional[int], info: str) -> str:
    """Append momentum signal to the investment info."""
    if momentum is None:
        return info

    if momentum == -1:
        info += "The cumulative return of past 3 days for this stock is NEGATIVE.\n"
    elif momentum == 0:
        info += "The cumulative return of past 3 days for this stock is ZERO.\n"
    elif momentum == 1:
        info += "The cumulative return of past 3 days for this stock is POSITIVE.\n"

    return info


def _parse_reflection_response(response: str) -> Dict[str, Any]:
    """Parse the LLM's reflection response into a structured dict."""
    response = response.strip()

    # Remove markdown code fences if present
    if response.startswith("```"):
        lines = response.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)

    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse reflection JSON: {response[:200]}")
        result: Dict[str, Any] = {}

        if "buy" in response.lower():
            result["investment_decision"] = "buy"
        elif "sell" in response.lower():
            result["investment_decision"] = "sell"
        else:
            result["investment_decision"] = "hold"

        result["summary_reason"] = response[:500]
        result["confidence"] = 0.5
        result["short_memory_ids"] = []
        result["mid_memory_ids"] = []
        result["long_memory_ids"] = []
        result["reflection_memory_ids"] = []
        return result


def _apply_promotion_bonus(
    brain: BrainDB,
    symbol: str,
    reflection_result: Dict[str, Any],
) -> None:
    """Apply importance bonus to pivotal memories identified by the LLM.
    
    Paper: Guardrails AI monitors memory IDs. Events identified as
    "pivotal for investment success" receive a +5 point bonus.
    On our [0,1] scale, that's +0.05.
    
    Args:
        brain: BrainDB memory system.
        symbol: Stock ticker.
        reflection_result: Parsed reflection containing memory IDs.
    """
    # Collect all memory IDs the LLM identified as influential
    pivotal_ids = []
    for key in ["short_memory_ids", "mid_memory_ids",
                "long_memory_ids", "reflection_memory_ids"]:
        ids = reflection_result.get(key, [])
        if isinstance(ids, list):
            pivotal_ids.extend([int(i) for i in ids if isinstance(i, (int, float))])
    
    if not pivotal_ids:
        return
    
    boosted = 0
    for mem_id in pivotal_ids:
        if brain.boost_importance(symbol, mem_id, bonus=0.05):
            boosted += 1
    
    if boosted > 0:
        logger.info(f"[{symbol}] Boosted {boosted} pivotal memories (+0.05 importance)")


def trading_reflection(
    cur_date: Union[date, datetime],
    symbol: str,
    brain: BrainDB,
    llm: LLMClient,
    character_string: str,
    top_k: int,
    run_mode: str,
    future_record: Optional[float] = None,
    momentum: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform a reflection step — the core working memory operation.

    In train mode: Given the actual future price change, reflect on which
    memories were most predictive.

    In test mode: Query memories and make a buy/hold/sell decision.

    Args:
        cur_date: Current trading date.
        symbol: Stock ticker.
        brain: The BrainDB memory system.
        llm: LLM client for generating reflections.
        character_string: Agent's dynamic character description (used as query).
        top_k: Number of memories to retrieve per layer.
        run_mode: "train" or "test".
        future_record: Next-day price change (train mode only).
        momentum: 3-day momentum signal: -1, 0, or 1 (test mode only).

    Returns:
        Dictionary with reflection results.
    """
    # Query all 4 memory layers using character string as the query
    short_mem, short_ids = brain.query_short(character_string, top_k, symbol)
    mid_mem, mid_ids = brain.query_mid(character_string, top_k, symbol)
    long_mem, long_ids = brain.query_long(character_string, top_k, symbol)
    refl_mem, refl_ids = brain.query_reflection(character_string, top_k, symbol)

    logger.info(f"[{symbol}] Queried memories — short:{len(short_mem)}, "
                f"mid:{len(mid_mem)}, long:{len(long_mem)}, reflection:{len(refl_mem)}")

    # Format memories for prompt
    memory_text = _format_memories_for_prompt(
        short_mem, short_ids, mid_mem, mid_ids,
        long_mem, long_ids, refl_mem, refl_ids,
    )

    # Build the investment info
    if run_mode == "train":
        investment_info = train_investment_info_prefix.format(
            cur_date=cur_date, symbol=symbol, future_record=future_record
        )
        investment_info += memory_text
        prompt = train_prompt.format(investment_info=investment_info)
    else:
        investment_info = test_investment_info_prefix.format(
            cur_date=cur_date, symbol=symbol
        )
        investment_info += memory_text

        if short_mem:
            investment_info += test_sentiment_explanation

        if momentum is not None:
            investment_info += test_momentum_explanation
            investment_info = _add_momentum_info(momentum, investment_info)

        prompt = test_prompt.format(investment_info=investment_info)

    # Call LLM
    try:
        response = llm.chat(prompt)
        result = _parse_reflection_response(response)
    except Exception as e:
        logger.error(f"Reflection LLM call failed: {e}")
        result = {
            "summary_reason": f"Reflection failed: {str(e)}",
            "short_memory_ids": [],
            "mid_memory_ids": [],
            "long_memory_ids": [],
            "reflection_memory_ids": [],
        }
        if run_mode == "test":
            result["investment_decision"] = "hold"
            result["confidence"] = 0.0

    # Store reflection summary in reflection memory
    if "summary_reason" in result and result["summary_reason"]:
        brain.add_memory_reflection(
            symbol=symbol,
            mem_date=cur_date,
            text=f"[{cur_date}] Reflection: {result['summary_reason'][:500]}",
        )

    # Paper: Apply promotion bonus to pivotal memories
    _apply_promotion_bonus(brain, symbol, result)

    # Build full result with all memory IDs used
    result["_all_memory_ids"] = {
        "short": short_ids,
        "mid": mid_ids,
        "long": long_ids,
        "reflection": refl_ids,
    }

    logger.info(f"[{symbol}] Reflection result: "
                f"{'decision=' + result.get('investment_decision', 'N/A') + ', ' if run_mode == 'test' else ''}"
                f"reason={result.get('summary_reason', '')[:100]}...")

    return result
