#!/usr/bin/env python3
"""
FinMEM Trading Agent - Entry Point

A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design.
Based on the research paper: https://arxiv.org/abs/2311.13743
"""

import argparse
import logging
from datetime import date, datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finmem.simulation.simulator import TradingSimulator
from finmem.config import FinMEMConfig


def main():
    parser = argparse.ArgumentParser(
        description="FinMEM Trading Agent - LLM-based trading with layered memory"
    )
    
    parser.add_argument(
        "--ticker", "-t",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)"
    )
    
    parser.add_argument(
        "--start-date", "-s",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: 30 days ago"
    )
    
    parser.add_argument(
        "--end-date", "-e",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Mode: train (populate memory + reflect) or test (make decisions)"
    )
    
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=100000.0,
        help="Initial capital (default: 100000)"
    )
    
    parser.add_argument(
        "--risk",
        type=str,
        choices=["conservative", "moderate", "aggressive"],
        default="moderate",
        help="Risk profile (default: moderate)"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Path to pre-built dataset pickle (optional)"
    )
    
    parser.add_argument(
        "--checkpoint", "-ckp",
        type=str,
        default=None,
        help="Path to load checkpoint from (resumes previous run)"
    )
    
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Path to save checkpoint after run"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Cognitive span: memories per layer to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (less output)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
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
    
    risk_map = {
        "conservative": 0.3,
        "moderate": 0.5,
        "aggressive": 0.7,
    }
    config.profile.risk_tolerance = risk_map[args.risk]
    
    # Create simulator (from checkpoint or fresh)
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
    
    # Save checkpoint if requested
    if args.save_checkpoint:
        simulator.save_checkpoint(args.save_checkpoint)
        print(f"  💾 Checkpoint saved to {args.save_checkpoint}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  📊 Results Summary")
    print(f"{'='*60}")
    print(f"  Period:        {result.start_date} → {result.end_date}")
    print(f"  Days:          {result.days_processed}")
    print(f"  Mode:          {result.mode}")
    print(f"  Initial:       ${result.initial_capital:,.2f}")
    print(f"  Final:         ${result.final_value:,.2f}")
    print(f"  Return:        ${result.total_return:,.2f} ({result.total_return_pct:+.2f}%)")
    
    # Paper's 5 evaluation metrics
    if result.metrics:
        print(f"\n  📈 Paper Metrics (FinMEM vs Buy & Hold):")
        print(f"  {'─'*50}")
        m = result.metrics
        bh = result.bh_metrics
        print(f"  {'Metric':<25} {'FinMEM':>12} {'B&H':>12}")
        print(f"  {'Cum. Return (%)':<25} {m.get('cumulative_return_pct', 0):>11.2f}% {bh.get('cumulative_return_pct', 0):>11.2f}%")
        print(f"  {'Sharpe Ratio':<25} {m.get('sharpe_ratio', 0):>12.4f} {bh.get('sharpe_ratio', 0):>12.4f}")
        print(f"  {'Ann. Volatility':<25} {m.get('annualized_volatility', 0):>12.4f} {bh.get('annualized_volatility', 0):>12.4f}")
        print(f"  {'Daily Volatility':<25} {m.get('daily_volatility', 0):>12.6f} {bh.get('daily_volatility', 0):>12.6f}")
        print(f"  {'Max Drawdown (%)':<25} {m.get('max_drawdown_pct', 0):>11.2f}% {bh.get('max_drawdown_pct', 0):>11.2f}%")
    
    print(f"\n  Memory Stats:  {result.memory_stats}")
    
    # Show last few trades
    trades = [t for t in result.trades if t.get("action") != "HOLD"]
    if trades:
        print(f"\n  💰 Trades ({len(trades)} total):")
        for trade in trades[-5:]:
            print(f"    {trade.get('date')} | {trade.get('action')} "
                  f"{trade.get('shares_traded', 0):.2f} shares @ ${trade.get('price', 0):.2f}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
