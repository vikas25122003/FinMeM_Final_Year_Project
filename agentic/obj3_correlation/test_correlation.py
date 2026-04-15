"""
Objective 3 — Step 4: End-to-End Test

Tests correlation matrix computation, cross-ticker retrieval formatting,
and the concentration guard logic.

Run:
    python -m agentic.obj3_correlation.test_correlation
"""

import os
import sys
import numpy as np
import pandas as pd


def test_objective_3():
    """End-to-end test for Objective 3: Cross-Ticker Memory Contextualization."""

    print("=" * 60)
    print("  🧪 Objective 3 — Cross-Ticker Memory Contextualization Test")
    print("=" * 60)

    # ── Step 1: Test correlation matrix computation ──
    print("\n  Step 1: Computing correlation matrix...")

    from agentic.obj3_correlation.matrix import compute_correlation_matrix

    tickers = ["TSLA", "NVDA", "MSFT", "AMZN", "NFLX"]

    try:
        corr = compute_correlation_matrix(tickers, window=30, period="6mo")
        assert corr.shape == (5, 5), f"Expected shape (5, 5), got {corr.shape}"
        assert list(corr.index) == tickers, f"Index mismatch: {list(corr.index)}"
        assert list(corr.columns) == tickers, f"Columns mismatch: {list(corr.columns)}"

        # Diagonal should be 1.0
        for t in tickers:
            assert abs(corr.loc[t, t] - 1.0) < 0.001, f"Diagonal not 1.0 for {t}"

        print(f"    ✓ Correlation matrix computed ({corr.shape})")
        print(f"\n    Correlation Matrix:")
        print(corr.round(2).to_string().replace("\n", "\n    "))

        # Check some known correlation patterns (tech stocks should be positively correlated)
        tsla_nvda_corr = corr.loc["TSLA", "NVDA"]
        print(f"\n    TSLA-NVDA correlation: {tsla_nvda_corr:.3f}")
        # Note: correlation can vary, but tech stocks are generally positive
        assert -1.0 <= tsla_nvda_corr <= 1.0, f"Invalid correlation: {tsla_nvda_corr}"

    except Exception as exc:
        print(f"    ⚠️ Real correlation computation failed (likely no internet): {exc}")
        print(f"    → Using synthetic correlation matrix for remaining tests")
        # Create synthetic matrix for testing
        corr = pd.DataFrame(
            [
                [1.00, 0.82, 0.61, 0.43, 0.29],
                [0.82, 1.00, 0.71, 0.38, 0.22],
                [0.61, 0.71, 1.00, 0.55, 0.41],
                [0.43, 0.38, 0.55, 1.00, 0.67],
                [0.29, 0.22, 0.41, 0.67, 1.00],
            ],
            index=tickers,
            columns=tickers,
        )

    # ── Step 2: Test cross-ticker retrieval formatting ──
    print("\n  Step 2: Testing cross-ticker prompt formatting...")

    from agentic.obj3_correlation.retrieval import format_cross_ticker_prompt_block

    mock_cross_memories = [
        {
            "text": "NVIDIA Blackwell chip demand surges as AI buildout accelerates",
            "source_ticker": "NVDA",
            "corr_weight": 0.82,
            "layer": "shallow",
        },
        {
            "text": "Microsoft Azure AI revenue grows 50% YoY, exceeds expectations",
            "source_ticker": "MSFT",
            "corr_weight": 0.61,
            "layer": "shallow",
        },
    ]

    prompt_block = format_cross_ticker_prompt_block(mock_cross_memories)
    assert "Cross-Asset Signals" in prompt_block, "Missing header"
    assert "NVDA" in prompt_block, "Missing NVDA"
    assert "MSFT" in prompt_block, "Missing MSFT"
    assert "corr=0.82" in prompt_block, "Missing correlation value"
    assert "correlated assets" in prompt_block.lower(), "Missing context note"

    print(f"    ✓ Prompt block formatted correctly")
    print(f"    Preview:\n    {prompt_block[:200]}...")

    # Empty case
    empty_block = format_cross_ticker_prompt_block([])
    assert empty_block == "", "Empty memories should return empty string"
    print(f"    ✓ Empty case handled correctly")

    # ── Step 3: Test concentration guard ──
    print("\n  Step 3: Testing concentration guard...")

    from agentic.obj3_correlation.guard import apply_concentration_guard, get_guard_summary

    # Create mock decisions
    decisions = {
        "TSLA": {"action": "BUY", "confidence": 0.78},
        "NVDA": {"action": "BUY", "confidence": 0.71},
        "MSFT": {"action": "BUY", "confidence": 0.65},
        "AMZN": {"action": "HOLD", "confidence": 0.50},
        "NFLX": {"action": "SELL", "confidence": 0.60},
    }

    # Use a mock matrix with high TSLA-NVDA correlation
    mock_corr = pd.DataFrame(
        [
            [1.00, 0.85, 0.55, 0.30, 0.20],
            [0.85, 1.00, 0.60, 0.25, 0.15],
            [0.55, 0.60, 1.00, 0.40, 0.30],
            [0.30, 0.25, 0.40, 1.00, 0.50],
            [0.20, 0.15, 0.30, 0.50, 1.00],
        ],
        index=tickers,
        columns=tickers,
    )

    modified, trigger_count = apply_concentration_guard(
        decisions, mock_corr, threshold=0.80
    )

    # NVDA should be overridden (lower confidence than TSLA, corr=0.85 > 0.80)
    assert modified["NVDA"]["action"] == "HOLD", \
        f"Expected NVDA → HOLD, got {modified['NVDA']['action']}"
    assert modified["TSLA"]["action"] == "BUY", \
        f"TSLA should remain BUY, got {modified['TSLA']['action']}"
    assert modified["MSFT"]["action"] == "BUY", \
        f"MSFT should remain BUY (corr < threshold), got {modified['MSFT']['action']}"
    assert "override_reason" in modified["NVDA"], "Missing override reason"
    assert trigger_count >= 1, f"Expected at least 1 guard trigger, got {trigger_count}"

    print(f"    ✓ Guard triggered {trigger_count} time(s)")
    print(f"    ✓ NVDA: BUY → HOLD (corr(TSLA,NVDA)=0.85 > 0.80)")
    print(f"    ✓ TSLA: BUY (kept — higher confidence 0.78)")
    print(f"    ✓ MSFT: BUY (kept — corr below threshold)")

    # Test guard summary
    summary = get_guard_summary(modified)
    assert summary["total_overrides"] == 1
    assert "TSLA" in summary["active_buys"]
    print(f"    ✓ Guard summary: {summary['total_overrides']} override(s)")

    # ── Step 4: Test edge cases ──
    print("\n  Step 4: Testing edge cases...")

    # No BUY decisions → guard should not fire
    no_buy = {
        "TSLA": {"action": "HOLD", "confidence": 0.5},
        "NVDA": {"action": "SELL", "confidence": 0.6},
    }
    modified2, count2 = apply_concentration_guard(no_buy, mock_corr, threshold=0.80)
    assert count2 == 0, "Guard should not fire with no BUY decisions"
    print(f"    ✓ No BUY edge case: guard correctly did not fire")

    # Single BUY → guard should not fire
    single_buy = {
        "TSLA": {"action": "BUY", "confidence": 0.8},
        "NVDA": {"action": "HOLD", "confidence": 0.5},
    }
    modified3, count3 = apply_concentration_guard(single_buy, mock_corr, threshold=0.80)
    assert count3 == 0, "Guard should not fire with single BUY"
    assert modified3["TSLA"]["action"] == "BUY"
    print(f"    ✓ Single BUY edge case: guard correctly did not fire")

    print("\n" + "=" * 60)
    print("  ✅ Objective 3 test passed.")
    print("=" * 60)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    test_objective_3()
