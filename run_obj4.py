#!/usr/bin/env python3
"""
Objective 4 Runner — Multi-Agent Memory Architecture

Runs the full Obj1+2+3+4 pipeline:
    LangGraph orchestrates 4 specialist agents (using TradingAgents prompts)
    each powered by a different AWS Bedrock model, each with dedicated
    FinMEM memory layers.

Usage:
    # Test the pipeline with mock data (no LLM calls)
    python run_obj4.py --test-obj4

    # Run full simulation on a ticker
    python run_obj4.py --ticker TSLA --mode test \\
        --start-date 2022-10-03 --end-date 2022-12-28

    # Show the LangGraph Mermaid diagram
    python run_obj4.py --show-graph

    # Run single-day multi-agent decision
    python run_obj4.py --ticker TSLA --single-day 2024-03-15

Environment:
    All objectives are enabled: ADAPTIVE_Q, LEARNED_IMPORTANCE,
    CROSS_TICKER, MULTIAGENT.
"""

import os
import sys
import argparse
import logging
from datetime import date, datetime, timedelta

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Force all objectives ON for Obj4
os.environ.setdefault("ADAPTIVE_Q", "true")
os.environ.setdefault("LEARNED_IMPORTANCE", "true")
os.environ.setdefault("CROSS_TICKER", "true")
os.environ.setdefault("MULTIAGENT", "true")
os.environ.setdefault("DEBATE_ROUNDS", "2")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_obj4")


def run_test():
    """Run the Objective 4 test suite (no LLM calls)."""
    logger.info("Running Objective 4 test suite...")
    from agentic.obj4_multiagent.test_multiagent import main as test_main
    return test_main()


def show_graph():
    """Print the LangGraph Mermaid diagram."""
    from agentic.obj4_multiagent.graph import get_graph_diagram
    print("\n📊 LangGraph Multi-Agent Trading Pipeline\n")
    print(get_graph_diagram())


def run_single_day(ticker: str, trade_date: str):
    """Run a single-day multi-agent decision."""
    import yfinance as yf
    from agentic.obj4_multiagent.graph import build_graph

    logger.info(f"Single-day run: {ticker} @ {trade_date}")

    # Fetch price history
    end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=90)
    hist = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"),
                       end=end_dt.strftime("%Y-%m-%d"), progress=False)
    if hist.empty:
        logger.error(f"No price data for {ticker}")
        return

    prices = hist["Close"].values.flatten().tolist()[-60:]
    cur_price = prices[-1] if prices else 0.0

    # Build graph
    graph = build_graph()

    # Create initial state (no BrainDB — cold start)
    initial_state = {
        "ticker": ticker,
        "cur_date": trade_date,
        "cur_price": cur_price,
        "price_history": prices,
        "brain": None,  # No memory system for single-day run
        "run_mode": "test",
        "portfolio_state": {
            "cash": 100000,
            "shares": 0,
            "position_value": 0,
            "total_value": 100000,
        },
        "character_string": f"trading analyst evaluating {ticker}",
        "top_k": 5,
        "risk_notes": [],
        "all_pivotal_ids": [],
    }

    logger.info("Invoking LangGraph pipeline...")
    result = graph.invoke(initial_state)

    # Print results
    print("\n" + "═" * 60)
    print(f"  MULTI-AGENT DECISION: {ticker} @ {trade_date}")
    print("═" * 60)
    print(f"  Regime:        {result.get('regime', '?')}")
    print(f"  Fundamental:   {result.get('fundamental_report', {}).get('direction', '?')} "
          f"(conf: {result.get('fundamental_report', {}).get('confidence', 0):.2f})")
    print(f"  Sentiment:     {result.get('sentiment_report', {}).get('direction', '?')} "
          f"(conf: {result.get('sentiment_report', {}).get('confidence', 0):.2f})")
    print(f"  Technical:     {result.get('technical_report', {}).get('direction', '?')} "
          f"(conf: {result.get('technical_report', {}).get('confidence', 0):.2f})")
    print(f"  Debate:        {result.get('debate_state', {}).get('consensus_direction', '?')} "
          f"(conf: {result.get('debate_state', {}).get('consensus_confidence', 0):.2f})")
    print(f"  Risk Notes:    {result.get('risk_notes', [])}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  FINAL DECISION: {result.get('final_decision', '?')}")
    print(f"  Confidence:    {result.get('final_confidence', 0):.2f}")
    print(f"  Kelly Shares:  {result.get('kelly_shares', 0)}")
    print(f"  Rationale:     {result.get('final_rationale', 'N/A')[:200]}")
    print("═" * 60)


def run_simulation(ticker: str, mode: str, start_date: str, end_date: str):
    """Run full simulation with multi-agent decisions."""
    import yfinance as yf
    from finmem.memory.layered_memory import BrainDB
    from agentic.obj4_multiagent.graph import build_graph

    logger.info(f"Simulation: {ticker} | {mode} | {start_date} → {end_date}")

    # Fetch price data
    hist = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if hist.empty:
        logger.error(f"No price data for {ticker}")
        return

    # Initialize BrainDB with paper-faithful defaults
    brain = BrainDB.create_default()

    # Build graph
    graph = build_graph()

    # Portfolio tracking
    cash = 100000.0
    shares = 0
    decisions = []

    dates = hist.index.tolist()
    prices_all = hist["Close"].values.flatten().tolist()

    for i, (trade_date, cur_price) in enumerate(zip(dates, prices_all)):
        date_str = trade_date.strftime("%Y-%m-%d") if hasattr(trade_date, 'strftime') else str(trade_date)[:10]

        # Price history (last 60 days)
        start_idx = max(0, i - 60)
        price_history = prices_all[start_idx:i + 1]

        if len(price_history) < 5:
            continue

        position_value = shares * cur_price
        total_value = cash + position_value

        initial_state = {
            "ticker": ticker,
            "cur_date": date_str,
            "cur_price": float(cur_price),
            "price_history": price_history,
            "brain": brain,
            "run_mode": mode,
            "portfolio_state": {
                "cash": cash,
                "shares": shares,
                "position_value": position_value,
                "total_value": total_value,
            },
            "character_string": f"trading analyst evaluating {ticker}",
            "top_k": 5,
            "risk_notes": [],
            "all_pivotal_ids": [],
        }

        try:
            result = graph.invoke(initial_state)
            decision = result.get("final_decision", "HOLD")
            kelly = result.get("kelly_shares", 0)

            # Execute trade
            if decision == "BUY" and kelly > 0:
                cost = kelly * cur_price
                if cost <= cash:
                    cash -= cost
                    shares += kelly
                    logger.info(f"[{date_str}] BUY {kelly} @ ${cur_price:.2f}")

            elif decision == "SELL" and shares > 0:
                sell_shares = min(abs(kelly) if kelly < 0 else shares, shares)
                cash += sell_shares * cur_price
                shares -= sell_shares
                logger.info(f"[{date_str}] SELL {sell_shares} @ ${cur_price:.2f}")

            else:
                logger.info(f"[{date_str}] HOLD (regime: {result.get('regime', '?')})")

            decisions.append({
                "date": date_str,
                "price": float(cur_price),
                "decision": decision,
                "confidence": result.get("final_confidence", 0),
                "regime": result.get("regime", "?"),
                "shares": shares,
                "cash": cash,
                "total_value": cash + shares * cur_price,
            })

        except Exception as e:
            logger.error(f"[{date_str}] Pipeline failed: {e}")
            decisions.append({
                "date": date_str,
                "decision": "ERROR",
                "error": str(e),
            })

    # Final summary
    final_value = cash + shares * prices_all[-1]
    cr = (final_value - 100000) / 100000 * 100
    bh_cr = (prices_all[-1] - prices_all[0]) / prices_all[0] * 100

    print("\n" + "═" * 60)
    print(f"  SIMULATION RESULTS: {ticker} ({start_date} → {end_date})")
    print("═" * 60)
    print(f"  Initial:       $100,000.00")
    print(f"  Final:         ${final_value:,.2f}")
    print(f"  CR:            {cr:+.2f}%")
    print(f"  Buy & Hold CR: {bh_cr:+.2f}%")
    print(f"  Total Days:    {len(decisions)}")
    print(f"  Decisions:     {sum(1 for d in decisions if d.get('decision') == 'BUY')} BUY | "
          f"{sum(1 for d in decisions if d.get('decision') == 'SELL')} SELL | "
          f"{sum(1 for d in decisions if d.get('decision') == 'HOLD')} HOLD")
    print("═" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Objective 4 — Multi-Agent Memory Architecture Runner"
    )
    parser.add_argument("--test-obj4", action="store_true",
                       help="Run test suite (no LLM calls)")
    parser.add_argument("--show-graph", action="store_true",
                       help="Print LangGraph Mermaid diagram")
    parser.add_argument("--ticker", type=str, default="TSLA",
                       help="Stock ticker (default: TSLA)")
    parser.add_argument("--mode", type=str, default="test",
                       choices=["train", "test"],
                       help="Simulation mode")
    parser.add_argument("--start-date", type=str, default="2022-10-03")
    parser.add_argument("--end-date", type=str, default="2022-12-28")
    parser.add_argument("--single-day", type=str, default=None,
                       help="Run single-day decision (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.test_obj4:
        sys.exit(run_test())
    elif args.show_graph:
        show_graph()
    elif args.single_day:
        run_single_day(args.ticker, args.single_day)
    else:
        run_simulation(args.ticker, args.mode, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
