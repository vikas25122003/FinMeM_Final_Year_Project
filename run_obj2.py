#!/usr/bin/env python3
"""
run_obj2.py — FinMEM + Objective 1 + Objective 2

Runs FinMEM with both Adaptive Q (Obj1) and Learned Importance (Obj2).

Usage:
    # Train (populate memory + collect reflection logs)
    python run_obj2.py --ticker TSLA --mode train \
        --start-date 2022-03-14 --end-date 2022-06-15

    # Train the importance classifier from collected logs
    python run_obj2.py --train-classifier --ticker TSLA

    # Test with learned importance
    python run_obj2.py --ticker TSLA --mode test \
        --start-date 2022-06-16 --end-date 2022-12-28
"""

import os

# ── Enable Objective 1 + 2 ────────────────────────────────────────────────
os.environ["ADAPTIVE_Q"] = "true"
os.environ["LEARNED_IMPORTANCE"] = "true"

print("=" * 60)
print("  🎯 Objective 1 + 2 ACTIVE")
print("  • Obj1: Regime-Conditioned Adaptive Q")
print("  • Obj2: Learned Importance Scoring")
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
        description="FinMEM + Obj1 + Obj2: Adaptive Q + Learned Importance"
    )

    parser.add_argument("--ticker", "-t", type=str, default="TSLA")
    parser.add_argument("--start-date", "-s", type=str, default=None)
    parser.add_argument("--end-date", "-e", type=str, default=None)
    parser.add_argument("--mode", "-m", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--capital", "-c", type=float, default=100000.0)
    parser.add_argument("--risk", type=str, choices=["conservative", "moderate", "aggressive"], default="moderate")
    parser.add_argument("--dataset", "-d", type=str, default=None)
    parser.add_argument("--checkpoint", "-ckp", type=str, default=None)
    parser.add_argument("--save-checkpoint", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    # Objective 2 specific
    parser.add_argument("--train-classifier", action="store_true",
                        help="Train the importance classifier from collected reflection logs")
    parser.add_argument("--test-obj2", action="store_true",
                        help="Run Objective 2 unit test")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handle Obj2-specific commands
    if args.test_obj2:
        from agentic.obj2_importance.test_importance import test_objective_2
        test_objective_2()
        return

    if args.train_classifier:
        from agentic.obj2_importance.trainer import run_training_pipeline
        run_training_pipeline(args.ticker)
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

    # Create simulator
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n  📦 Loading checkpoint from {args.checkpoint}...")
        simulator = TradingSimulator.load_checkpoint(args.checkpoint)
    else:
        simulator = TradingSimulator(config)

    # Run simulation
    result = simulator.run(
        ticker=args.ticker,
        start_date=start_date,
        end_date=end_date,
        mode=args.mode,
        initial_capital=args.capital,
        dataset_path=args.dataset,
        verbose=not args.quiet,
    )

    # Save checkpoint
    if args.save_checkpoint:
        simulator.save_checkpoint(args.save_checkpoint)
        print(f"\n  💾 Checkpoint saved → {args.save_checkpoint}/")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  📊 Results — Obj1 (Adaptive Q) + Obj2 (Learned Importance)")
    print(f"{'='*60}")
    print(f"  Period:  {result.start_date} → {result.end_date}")
    print(f"  Days:    {result.days_processed}")
    print(f"  Return:  ${result.total_return:,.2f} ({result.total_return_pct:+.2f}%)")

    if result.metrics:
        m = result.metrics
        bh = result.bh_metrics
        print(f"\n  {'Metric':<25} {'FinMEM+Obj1+2':>12} {'B&H':>12}")
        print(f"  {'─'*50}")
        print(f"  {'Cum. Return (%)':25} {m.get('cumulative_return_pct', 0):>11.2f}% {bh.get('cumulative_return_pct', 0):>11.2f}%")
        print(f"  {'Sharpe Ratio':25} {m.get('sharpe_ratio', 0):>12.4f} {bh.get('sharpe_ratio', 0):>12.4f}")
        print(f"  {'Max Drawdown (%)':25} {m.get('max_drawdown_pct', 0):>11.2f}% {bh.get('max_drawdown_pct', 0):>11.2f}%")

    # Obj2: show importance model info
    try:
        from agentic.obj2_importance.inference import get_model_info
        info = get_model_info()
        print(f"\n  📦 Importance Model: {info}")
    except Exception:
        pass

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
