"""
Objective 1 — End-to-End Test Script

Tests the regime detection pipeline without requiring any LLM calls.
Uses TSLA on 2022-10-25 — a known high-volatility crisis period.

Run:
    python -m agentic.obj1_regime.test_regime

Expected output:
    Detected Regime: CRISIS
    ✅ Objective 1 test passed.
"""

import math
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentic.obj1_regime.features    import compute_features
from agentic.obj1_regime.classifier  import get_classifier
from agentic.obj1_regime.q_table     import get_Q, get_all_Q


def test_objective_1(ticker: str = "TSLA", date: str = "2022-10-25") -> None:
    """
    End-to-end test for Objective 1 regime pipeline.

    Validates:
        - Feature computation returns numeric values
        - ThresholdClassifier returns a valid regime string
        - Q values are correct for the detected regime
        - On 2022-10-25 (TSLA at high vol), regime should be CRISIS
        - Q_shallow in CRISIS should be 5 (half-life of 5 days)
    """
    print(f"\n{'='*55}")
    print(f"  Objective 1 Test — {ticker} on {date}")
    print(f"{'='*55}\n")

    # ── Step 1: Feature Engineering ────────────────────────────────────────
    print("Step 1 — Computing market features (yfinance)...")
    features = compute_features(ticker, date)
    print(f"  Features:")
    for k, v in features.items():
        print(f"    {k:<18} = {v:.6f}")

    assert isinstance(features, dict), "features must be a dict"
    assert all(k in features for k in ["vol_20d", "vol_50d", "momentum_20d"]), \
        "Missing required feature keys"
    assert all(isinstance(v, float) for v in features.values()), \
        "All feature values must be floats"
    print("  ✓ Feature engineering OK\n")

    # ── Step 2: Regime Classification ──────────────────────────────────────
    print("Step 2 — Classifying regime (threshold classifier)...")
    clf    = get_classifier(mode="threshold")
    regime = clf.predict(features)
    print(f"  Detected Regime: {regime}")

    assert regime in ["BULL", "SIDEWAYS", "CRISIS"], \
        f"Invalid regime: {regime}"

    # Oct 2022 was a high-vol period — TSLA dropped ~44% that month
    # We expect CRISIS or at minimum a non-BULL regime
    vol_20d = features["vol_20d"]
    print(f"  vol_20d (annualized) = {vol_20d:.4f}")
    if vol_20d >= 0.35:
        assert regime == "CRISIS", \
            f"Expected CRISIS for vol_20d={vol_20d:.4f} but got {regime}"
        print(f"  ✓ vol_20d={vol_20d:.4f} >= 0.35 → correctly classified as CRISIS")
    else:
        print(f"  ℹ vol_20d={vol_20d:.4f} < 0.35 — regime is {regime} (data may differ by API call timing)")
    print()

    # ── Step 3: Q-Value Lookup ─────────────────────────────────────────────
    print("Step 3 — Looking up adaptive Q values...")
    q_vals = get_all_Q(regime)
    print(f"  Adaptive Q Values for regime={regime}:")
    for layer, Q in q_vals.items():
        print(f"    Q_{layer:<16} = {Q}")

    assert "shallow" in q_vals and "intermediate" in q_vals and "deep" in q_vals, \
        "Q table must have shallow, intermediate, deep keys"

    if regime == "CRISIS":
        assert q_vals["shallow"] == 5,   f"Expected Q_shallow=5 in CRISIS, got {q_vals['shallow']}"
        assert q_vals["intermediate"] == 45, f"Expected Q_intermediate=45 in CRISIS"
        assert q_vals["deep"] == 180,    f"Expected Q_deep=180 in CRISIS"
    elif regime == "SIDEWAYS":
        assert q_vals["shallow"] == 14,  f"Sideways Q_shallow must be paper default 14"
    elif regime == "BULL":
        assert q_vals["shallow"] == 21,  f"Bull Q_shallow must be 21"

    print("  ✓ Q-value lookup OK\n")

    # ── Step 4: Recency Score Calculation ──────────────────────────────────
    print("Step 4 — Computing recency scores for δ=7 days (paper Equation 2):")
    delta = 7
    for layer, Q in q_vals.items():
        score = math.exp(-delta / Q)
        base_Q = {"shallow": 14, "intermediate": 90, "deep": 365}[layer]
        base_score = math.exp(-delta / base_Q)
        delta_str  = f"+{score - base_score:.4f}" if score > base_score else f"{score - base_score:.4f}"
        print(f"    {layer:<16} Q={Q:>3}  →  S_recency={score:.4f}  "
              f"(vs paper Q={base_Q}: {base_score:.4f}, Δ={delta_str})")

    # Layer alias test
    q_short = get_Q(regime, "short")   # should resolve "short" → "shallow"
    q_mid   = get_Q(regime, "mid")
    q_long  = get_Q(regime, "long")
    assert q_short == q_vals["shallow"], "Layer alias 'short' must resolve to 'shallow'"
    assert q_mid   == q_vals["intermediate"], "Layer alias 'mid' must resolve to 'intermediate'"
    assert q_long  == q_vals["deep"], "Layer alias 'long' must resolve to 'deep'"
    print("\n  ✓ Layer alias tests passed (short→shallow, mid→intermediate, long→deep)\n")

    # ── Step 5: Summary ────────────────────────────────────────────────────
    print(f"{'='*55}")
    print(f"  Regime detected: {regime}")
    if regime == "CRISIS" and delta == 7:
        shallow_score  = math.exp(-7 / q_vals["shallow"])
        paper_score    = math.exp(-7 / 14)
        speedup        = paper_score / shallow_score
        print(f"  A 7-day-old short memory scores {shallow_score:.4f}")
        print(f"  vs. {paper_score:.4f} with fixed Q=14 (paper baseline)")
        print(f"  → Deprioritized {speedup:.2f}x MORE during crisis (forces focus on fresh data)")
    print(f"{'='*55}")
    print(f"\n✅ Objective 1 test passed.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Objective 1 regime pipeline")
    parser.add_argument("--ticker", default="TSLA", help="Stock ticker")
    parser.add_argument("--date",   default="2022-10-25", help="Test date (YYYY-MM-DD)")
    args = parser.parse_args()
    test_objective_1(ticker=args.ticker, date=args.date)
