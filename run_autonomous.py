#!/usr/bin/env python3
"""
FinMEM + TradingAgents — Autonomous Trading System

End-to-end pipeline:
    1. HMM Regime Detection (Obj1, trained on SPY)
    2. Layered Memory System (FinMEM BrainDB)
    3. Multi-Agent Decision Making (TradingAgents LangGraph + Gemini)
    4. CVaR Portfolio Optimization (replaces Pearson correlation)
    5. Paper Trading Execution (Alpaca with limit orders + stop-losses)
    6. Dual Reflection System (TradingAgents BM25 + FinMEM BrainDB)

Usage:
    # Single cycle (dry run — no real orders)
    python run_autonomous.py --dry-run

    # Single cycle with Alpaca paper execution
    python run_autonomous.py --single

    # Autonomous loop (runs during market hours)
    python run_autonomous.py --autonomous

    # Autonomous loop, dry run (no orders, for testing)
    python run_autonomous.py --autonomous --dry-run

    # Custom tickers
    PORTFOLIO_TICKERS=TSLA,NVDA,AAPL python run_autonomous.py --autonomous

    # Train HMM first (run once)
    python run_autonomous.py --train-hmm

    # Show system status
    python run_autonomous.py --status

Environment Variables:
    PORTFOLIO_TICKERS       Comma-separated tickers (default: TSLA,NVDA,MSFT,AMZN,AAPL)
    ALPACA_API_KEY          Alpaca paper trading API key
    ALPACA_SECRET_KEY       Alpaca paper trading secret key
    GOOGLE_API_KEY          Gemini API key for TradingAgents
    AWS_ACCESS_KEY_ID       AWS credentials for Bedrock (Kimi K2.5)
    AWS_SECRET_ACCESS_KEY   AWS credentials for Bedrock
    AWS_REGION              AWS region (default: us-east-1)
    TA_LLM_PROVIDER         LLM provider for TradingAgents (default: google)
    TA_DEEP_LLM             Deep thinking model (default: gemini-2.5-pro)
    TA_QUICK_LLM            Quick thinking model (default: gemini-2.5-flash)
    CYCLE_INTERVAL          Seconds between cycles (default: 1800 = 30 min)
    CVAR_CONFIDENCE         CVaR confidence level (default: 0.95)
    STOP_LOSS_PCT           Stop-loss distance (default: 0.05 = 5%)
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TradingAgents"))

os.environ.setdefault("ADAPTIVE_Q", "true")
os.environ.setdefault("ADAPTIVE_Q_MODE", "hmm")
os.environ.setdefault("LEARNED_IMPORTANCE", "true")
os.environ.setdefault("CROSS_TICKER", "true")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("autonomous")


def train_hmm():
    """Train the HMM regime classifier on SPY."""
    from agentic.obj1_regime.train_hmm import train
    train(
        ticker="SPY",
        start_date="2017-01-01",
        end_date="2022-03-13",
        n_states=3,
        n_iter=200,
        save_path="./models/hmm_regime.pkl",
    )


def show_status():
    """Show current system status."""
    from orchestrator.config import get_orchestrator_config

    config = get_orchestrator_config()

    print(f"\n{'='*60}")
    print(f"  AUTONOMOUS TRADING SYSTEM — STATUS")
    print(f"{'='*60}")
    print(f"  Time:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Tickers:        {config['portfolio_tickers']}")
    print(f"  LLM Provider:   {config['ta_llm_provider']}")
    print(f"  Deep LLM:       {config['ta_deep_think_llm']}")
    print(f"  Quick LLM:      {config['ta_quick_think_llm']}")
    print(f"  Bedrock Model:  {config['bedrock_model']}")
    print(f"  Cycle Interval: {config['cycle_interval_seconds']}s")
    print(f"  CVaR Confidence:{config['cvar_confidence']}")
    print(f"  Stop Loss:      {config['stop_loss_pct']*100:.1f}%")
    print(f"  Max Position:   {config['cvar_max_weight']*100:.1f}%")

    hmm_path = config["hmm_model_path"]
    imp_path = config["importance_model_path"]
    print(f"\n  HMM Model:      {'EXISTS' if os.path.exists(hmm_path) else 'NOT FOUND — run --train-hmm'}")
    print(f"  Importance Clf:  {'EXISTS' if os.path.exists(imp_path) else 'NOT TRAINED YET'}")

    state_dir = config["state_dir"]
    brain_exists = os.path.exists(os.path.join(state_dir, "brain", "short", "state_dict.pkl"))
    portfolio_exists = os.path.exists(os.path.join(state_dir, "portfolio.json"))
    print(f"  BrainDB State:  {'EXISTS' if brain_exists else 'FRESH START'}")
    print(f"  Portfolio State: {'EXISTS' if portfolio_exists else 'FRESH START'}")

    try:
        from orchestrator.alpaca_enhanced import get_account_info, is_market_open
        market_open = is_market_open()
        account = get_account_info()
        print(f"\n  Alpaca Status:   {account['status']}")
        print(f"  Market Open:     {market_open}")
        print(f"  Cash:            ${account['cash']:,.2f}")
        print(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
        print(f"  Equity:          ${account['equity']:,.2f}")
    except Exception as e:
        print(f"\n  Alpaca:          NOT CONFIGURED ({e})")

    print(f"{'='*60}\n")


def run_single_cycle(dry_run: bool = False):
    """Run a single trading cycle."""
    from orchestrator.config import get_orchestrator_config
    from orchestrator.orchestrator import AutonomousTrader

    config = get_orchestrator_config()
    trader = AutonomousTrader(config)
    trader.initialize()

    result = trader.run_cycle(dry_run=dry_run)
    return result


def run_autonomous_loop(dry_run: bool = False):
    """Run the autonomous trading loop."""
    from orchestrator.config import get_orchestrator_config
    from orchestrator.orchestrator import AutonomousTrader

    config = get_orchestrator_config()
    trader = AutonomousTrader(config)
    trader.initialize()
    trader.run_autonomous(dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="FinMEM + TradingAgents — Autonomous Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_autonomous.py --train-hmm          # Train HMM on SPY (run once)
    python run_autonomous.py --status              # Check system status
    python run_autonomous.py --dry-run             # Single cycle, no orders
    python run_autonomous.py --single              # Single cycle with execution
    python run_autonomous.py --autonomous          # Full autonomous loop
    python run_autonomous.py --autonomous --dry-run  # Loop without orders
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-hmm", action="store_true",
                      help="Train HMM regime classifier on SPY (5 years)")
    mode.add_argument("--status", action="store_true",
                      help="Show system status and configuration")
    mode.add_argument("--single", action="store_true",
                      help="Run a single trading cycle")
    mode.add_argument("--autonomous", action="store_true",
                      help="Run the autonomous trading loop")

    parser.add_argument("--dry-run", action="store_true",
                        help="Skip actual Alpaca order submission")

    args = parser.parse_args()

    if args.train_hmm:
        train_hmm()
    elif args.status:
        show_status()
    elif args.single:
        run_single_cycle(dry_run=args.dry_run)
    elif args.autonomous:
        run_autonomous_loop(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
