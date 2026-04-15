"""
Objective 3 — Step 3: Portfolio Concentration Guard

Prevents over-allocation to highly correlated positions by
overriding the lower-confidence BUY to HOLD when pairwise
correlation exceeds the threshold.
"""

import os
import logging
from itertools import combinations
from typing import Dict, Any, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def apply_concentration_guard(
    decisions: Dict[str, Dict[str, Any]],
    corr_matrix: pd.DataFrame,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Check all BUY pairs and override lower-confidence to HOLD if correlated.

    Args:
        decisions:   Dict of ticker → {action, confidence, ...}.
                     Example: {"TSLA": {"action": "BUY", "confidence": 0.78}, ...}
        corr_matrix: DataFrame of pairwise correlations.
        threshold:   Maximum allowed correlation between two BUY positions.
                     Defaults to CONCENTRATION_THRESHOLD env var (0.80).

    Returns:
        Tuple of (modified_decisions, guard_trigger_count).
    """
    if threshold is None:
        threshold = float(os.getenv("CONCENTRATION_THRESHOLD", "0.80"))

    guard_trigger_count = 0

    # Extract BUY tickers
    buy_tickers = [
        t for t, d in decisions.items()
        if d.get("action", "").upper() == "BUY"
    ]

    if len(buy_tickers) < 2:
        logger.debug("[Obj3] Guard: fewer than 2 BUY decisions — nothing to check")
        return decisions, 0

    logger.info(f"[Obj3] Guard checking {len(buy_tickers)} BUY positions: {buy_tickers}")

    # Track which tickers have already been overridden
    overridden = set()

    # Check all pairs of BUY tickers
    for t1, t2 in combinations(buy_tickers, 2):
        # Skip if either already overridden
        if t1 in overridden or t2 in overridden:
            continue

        # Look up correlation
        try:
            corr_val = abs(float(corr_matrix.loc[t1, t2]))
        except (KeyError, ValueError):
            logger.debug(f"[Obj3] Guard: No correlation data for {t1}-{t2}")
            continue

        if corr_val > threshold:
            # Override the lower-confidence ticker
            conf1 = float(decisions[t1].get("confidence", 0.5))
            conf2 = float(decisions[t2].get("confidence", 0.5))

            if conf1 >= conf2:
                override_ticker = t2
                keep_ticker = t1
            else:
                override_ticker = t1
                keep_ticker = t2

            # Apply override
            decisions[override_ticker]["action"] = "HOLD"
            decisions[override_ticker]["override_reason"] = (
                f"Concentration guard: corr={corr_val:.2f} with {keep_ticker}"
            )
            decisions[override_ticker]["original_action"] = "BUY"

            overridden.add(override_ticker)
            guard_trigger_count += 1

            logger.info(
                f"[Obj3] Guard FIRED: {override_ticker} BUY→HOLD "
                f"(corr={corr_val:.2f} with {keep_ticker}, "
                f"conf {decisions[override_ticker].get('confidence', '?')} < {decisions[keep_ticker].get('confidence', '?')})"
            )

    if guard_trigger_count > 0:
        logger.info(f"[Obj3] Concentration guard triggered {guard_trigger_count} times")
    else:
        logger.debug("[Obj3] Guard: no concentration issues detected")

    return decisions, guard_trigger_count


def get_guard_summary(decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Get a summary of guard actions for logging/UI.

    Returns:
        Dict with counts and details of overrides.
    """
    overrides = []
    for ticker, d in decisions.items():
        if d.get("override_reason"):
            overrides.append({
                "ticker": ticker,
                "original_action": d.get("original_action", "BUY"),
                "new_action": d.get("action", "HOLD"),
                "reason": d.get("override_reason"),
            })

    return {
        "total_overrides": len(overrides),
        "overrides": overrides,
        "active_buys": [t for t, d in decisions.items() if d.get("action", "").upper() == "BUY"],
    }
