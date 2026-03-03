"""
Objective 1 — Step 3: Q-Value Lookup Table

Maps (market_regime, memory_layer) → Q stability value for recency decay.

Original FinMEM paper (Equation 2) uses fixed values:
    Q_shallow       = 14  days
    Q_intermediate  = 90  days
    Q_deep          = 365 days

We replace these with regime-dependent values. The SIDEWAYS row
preserves the paper's exact defaults, so disabling Objective 1
(ADAPTIVE_Q=false) produces identical results to the base paper.

Theoretical justification for Q values:
    CRISIS: Q_shallow=5  — in a crash, 5-year-old news is stale after 5 days.
            e.g., 2022 crypto crash, 2020 COVID: extreme vol lasts 2-3 weeks.
    BULL:   Q_shallow=21 — in a calm trend, positive earnings guidance lasts
            an entire month. Wider memory window helps sustain convictions.
    SIDEWAYS = paper defaults (exact match, no behavioural change).

Layer mapping to your codebase:
    "short"  → "shallow"       (Q=5/14/21)
    "mid"    → "intermediate"  (Q=45/90/120)
    "long"   → "deep"          (Q=180/365/400)

Mathematical impact example (delta=7 days):
    CRISIS  shallow: e^(-7/5)  = 0.247  (heavy decay — focus on very recent)
    SIDEWAYS shallow: e^(-7/14) = 0.607  (paper default)
    BULL    shallow: e^(-7/21) = 0.716  (slower decay — sustain old signals)
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


# ── Main Q-value lookup table ──────────────────────────────────────────────

Q_TABLE: Dict[str, Dict[str, int]] = {
    "BULL": {
        "shallow":      21,    # News stays relevant longer in calm bull markets
        "intermediate": 120,   # Quarterly trend signals persist through uptrend
        "deep":         400,   # Annual fundamentals extra stable during expansion
    },
    "SIDEWAYS": {
        "shallow":      14,    # ← Exact FinMEM paper defaults (preserved)
        "intermediate": 90,
        "deep":         365,
    },
    "CRISIS": {
        "shallow":      5,     # News expires very fast — last week is ancient history
        "intermediate": 45,    # Quarterly signals halved — regime flips quickly
        "deep":         180,   # Annual fundamentals lose half relevance in crash
    },
}

# Layer name aliases — map codebase names to Q_TABLE keys
_LAYER_ALIASES: Dict[str, str] = {
    # Internal FinMEM layer names → Q_TABLE layer keys
    "short":        "shallow",
    "shallow":      "shallow",
    "mid":          "intermediate",
    "intermediate": "intermediate",
    "middle":       "intermediate",
    "long":         "deep",
    "deep":         "deep",
    "reflection":   "shallow",   # Reflection uses shallow Q as conservative default
}


def get_Q(regime: str, layer: str) -> int:
    """
    Returns the Q stability value for recency decay.

    Args:
        regime: Market regime string. One of: BULL | SIDEWAYS | CRISIS
                Case-insensitive. Defaults to SIDEWAYS if unknown.
        layer:  Memory layer name. Accepts both internal names (short/mid/long)
                and Q_TABLE layer names (shallow/intermediate/deep).

    Returns:
        Q (int): Stability parameter in days used in e^(-delta / Q).

    Examples:
        get_Q("CRISIS", "shallow")          # 5
        get_Q("CRISIS", "short")            # 5   (alias)
        get_Q("BULL", "intermediate")       # 120
        get_Q("UNKNOWN_REGIME", "deep")     # 365 (falls back to SIDEWAYS)
    """
    regime_key = regime.upper().strip()
    if regime_key not in Q_TABLE:
        logger.warning(f"[Obj1] Unknown regime '{regime}' — defaulting to SIDEWAYS.")
        regime_key = "SIDEWAYS"

    layer_key = _LAYER_ALIASES.get(layer.lower().strip(), "shallow")
    if layer.lower().strip() not in _LAYER_ALIASES:
        logger.warning(f"[Obj1] Unknown layer '{layer}' — defaulting to shallow.")

    q_value = Q_TABLE[regime_key][layer_key]
    logger.debug(f"[Obj1] Q lookup: regime={regime_key}, layer={layer_key} → Q={q_value}")
    return q_value


def get_all_Q(regime: str) -> Dict[str, int]:
    """
    Returns all three Q values for a given regime.

    Args:
        regime: Market regime string.

    Returns:
        Dict with keys: shallow, intermediate, deep → Q int values.
    """
    regime_key = regime.upper().strip()
    if regime_key not in Q_TABLE:
        regime_key = "SIDEWAYS"
    return dict(Q_TABLE[regime_key])
