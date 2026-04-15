#!/usr/bin/env python3
"""
run_obj3.py — FinMEM + Objective 1 + 2 + 3

Runs FinMEM with all three objectives enabled:
  Obj1: Adaptive Q (Regime-Conditioned)
  Obj2: Learned Importance Scoring
  Obj3: Cross-Ticker Memory Contextualization

Usage:
    # Run full multi-ticker portfolio
    python run_obj3.py --tickers TSLA NVDA MSFT --mode test \
        --start-date 2022-10-01 --end-date 2023-04-25

    # Run correlation test
    python run_obj3.py --test-obj3

    # Run single ticker with cross-ticker awareness
    python run_obj3.py --ticker TSLA --mode test \
        --start-date 2022-10-01 --end-date 2023-04-25
"""

import os

# ── Enable all three objectives ───────────────────────────────────────────
os.environ["ADAPTIVE_Q"] = "true"
os.environ["LEARNED_IMPORTANCE"] = "true"
os.environ["CROSS_TICKER"] = "true"

print("=" * 60)
print("  🎯 Objective 1 + 2 + 3 ACTIVE")
print("  • Obj1: Regime-Conditioned Adaptive Q")
print("  • Obj2: Learned Importance Scoring")
print("  • Obj3: Cross-Ticker Memory + Concentration Guard")
print("=" * 60)
print()

import argparse
import logging
from datetime import date, datetime, timedelta
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finmem.simulation.simulator import TradingSimulator
from finmem.config import FinMEMConfig


def main():
    parser = argparse.ArgumentParser(
        description="FinMEM + Obj1 + Obj2 + Obj3: Full Agentic Pipeline"
    )

    parser.add_argument("--ticker", "-t", type=str, default="TSLA")
    parser.add_argument("--tickers", nargs="+", type=str, default=None,
                        help="Multiple tickers for portfolio mode (e.g., TSLA NVDA MSFT)")
    parser.add_argument("--start-date", "-s", type=str, default=None)
    parser.add_argument("--end-date", "-e", type=str, default=None)
    parser.add_argument("--mode", "-m", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--capital", "-c", type=float, default=100000.0)
    parser.add_argument("--risk", type=str, choices=["conservative", "moderate", "aggressive"], default="moderate")
    parser.add_argument("--dataset", "-d", type=str, default=None)
    parser.add_argument("--checkpoint", "-ckp", type=str, default=None)
    parser.add_argument("--save-checkpoint", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    # Objective 3 specific
    parser.add_argument("--test-obj3", action="store_true",
                        help="Run Objective 3 unit test")
    parser.add_argument("--show-correlation", action="store_true",
                        help="Display the correlation matrix and exit")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handle Obj3-specific commands
    if args.test_obj3:
        from agentic.obj3_correlation.test_correlation import test_objective_3
        test_objective_3()
        return

    if args.show_correlation:
        from agentic.obj3_correlation.matrix import compute_correlation_matrix
        tickers = args.tickers or os.getenv("PORTFOLIO_TICKERS", "TSLA,NVDA,MSFT,AMZN,NFLX").split(",")
        corr = compute_correlation_matrix([t.strip() for t in tickers])
        print("\n  📊 30-Day Rolling Correlation Matrix:")
        print(f"  {'─'*60}")
        print(corr.round(3).to_string().replace("\n", "\n  "))
        print()
        return

    # Parse dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end_date = date.today()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        start_date = end_date - timedelta(days=30)

    # Build config
    config = FinMEMConfig()
    config.initial_capital = args.capital
    config.memory.top_k = args.top_k

    risk_map = {"conservative": 0.3, "moderate": 0.5, "aggressive": 0.7}
    config.profile.risk_tolerance = risk_map[args.risk]

    # Multi-ticker or single-ticker
    tickers = args.tickers or [args.ticker]

    all_results = {}

    for ticker in tickers:
        print(f"\n{'─'*60}")
        print(f"  Running {ticker} ({args.mode} mode)...")
        print(f"{'─'*60}")

        # Create simulator
        if args.checkpoint and os.path.exists(args.checkpoint):
            simulator = TradingSimulator.load_checkpoint(args.checkpoint)
        else:
            simulator = TradingSimulator(config)

        result = simulator.run(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            mode=args.mode,
            initial_capital=args.capital / len(tickers),  # Equal allocation
            dataset_path=args.dataset,
            verbose=not args.quiet,
        )

        all_results[ticker] = result

        if args.save_checkpoint:
            ckp_path = f"{args.save_checkpoint}_{ticker}"
            simulator.save_checkpoint(ckp_path)

    # Portfolio-level summary
    if len(tickers) > 1:
        print(f"\n{'='*60}")
        print(f"  📊 Portfolio-Level Results (Obj1+2+3)")
        print(f"{'='*60}")
        print(f"\n  {'Ticker':8} {'Return ($)':>12} {'Return (%)':>12} {'Sharpe':>10}")
        print(f"  {'─'*50}")

        total_return = 0
        for t, r in all_results.items():
            total_return += r.total_return
            sharpe = r.metrics.get("sharpe_ratio", 0) if r.metrics else 0
            print(f"  {t:8} {r.total_return:>12,.2f} {r.total_return_pct:>11.2f}% {sharpe:>10.4f}")

        print(f"  {'─'*50}")
        total_capital = args.capital
        total_pct = (total_return / total_capital) * 100
        print(f"  {'TOTAL':8} {total_return:>12,.2f} {total_pct:>11.2f}%")

        # Show concentration guard summary
        print(f"\n  🛡️ Concentration Guard Summary:")
        print(f"  (Guard prevents over-allocation to correlated positions)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
