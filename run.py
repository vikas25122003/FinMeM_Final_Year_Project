#!/usr/bin/env python3
"""
FinMEM Trading Agent - Entry Point

A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design.
Based on the research paper: https://arxiv.org/abs/2311.13743
"""

import argparse
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finmem.simulation.simulator import TradingSimulator
from finmem.config import DEFAULT_CONFIG


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
        default="test",
        help="Mode: train (populate memory) or test (make decisions)"
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
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (less output)"
    )
    
    args = parser.parse_args()
    
    # Parse dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=30)
    
    # Update config
    config = DEFAULT_CONFIG
    config.initial_capital = args.capital
    
    risk_map = {
        "conservative": 0.3,
        "moderate": 0.5,
        "aggressive": 0.7
    }
    config.profile.risk_tolerance = risk_map[args.risk]
    
    # Create and run simulator
    print("\n" + "="*60)
    print("🧠 FinMEM Trading Agent")
    print("="*60)
    
    simulator = TradingSimulator(config)
    
    result = simulator.run(
        ticker=args.ticker,
        start_date=start_date,
        end_date=end_date,
        mode=args.mode,
        verbose=not args.quiet
    )
    
    # Print results
    print("\n" + "="*60)
    print("📊 Simulation Results")
    print("="*60)
    print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Value: ${result.final_value:,.2f}")
    print(f"Total Return: {result.total_return:.2f}%")
    print(f"Decisions Made: {len(result.decisions)}")
    print(f"Trades Executed: {len(result.trades)}")
    
    if result.decisions:
        print("\n📈 Latest Decision:")
        latest = result.decisions[-1]
        print(f"  Action: {latest.action.value}")
        print(f"  Confidence: {latest.confidence:.0%}")
        print(f"  Reasoning: {latest.reasoning[:200]}...")
    
    if result.trades:
        print("\n💰 Trades:")
        for trade in result.trades[-5:]:  # Show last 5
            print(f"  {trade['action']} {trade['shares']:.2f} shares @ ${trade['price']:.2f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
