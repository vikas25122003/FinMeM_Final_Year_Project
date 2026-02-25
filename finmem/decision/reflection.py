"""
Reflection Module

The core "working memory" mechanism from the FinMEM paper.
Queries all 4 memory layers, constructs a prompt, and uses the LLM
to either:
- Train mode: Reflect on what memories predicted the actual price change
- Test mode: Make a buy/hold/sell decision based on memories + momentum
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
    """Format retrieved memories into a prompt string.

    Args:
        *_memory: Lists of memory text content per layer.
        *_memory_ids: Corresponding memory IDs per layer.

    Returns:
        Formatted string with all memories labeled by layer and ID.
    """
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
    """Parse the LLM's reflection response into a structured dict.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed dictionary with reflection data.
    """
    response = response.strip()

    # Remove markdown code fences if present
    if response.startswith("```"):
        lines = response.split("\n")
        # Remove first and last lines (``` markers)
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
        # Fallback: try to extract key fields
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
        character_string: Agent's character description (used as query).
        top_k: Number of memories to retrieve per layer.
        run_mode: "train" or "test".
        future_record: Next-day price change (train mode only).
        momentum: 3-day momentum signal: -1, 0, or 1 (test mode only).

    Returns:
        Dictionary with reflection results:
        - Train: {summary_reason, short_memory_ids, mid_memory_ids, ...}
        - Test: {investment_decision, summary_reason, confidence, *_memory_ids}
    """
    # Query all 4 memory layers
    short_mem, short_ids = brain.query_short(character_string, top_k, symbol)
    mid_mem, mid_ids = brain.query_mid(character_string, top_k, symbol)
    long_mem, long_ids = brain.query_long(character_string, top_k, symbol)
    refl_mem, refl_ids = brain.query_reflection(character_string, top_k, symbol)

    # Log what was retrieved
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
