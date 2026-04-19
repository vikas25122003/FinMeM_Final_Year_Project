#!/usr/bin/env python3
"""
FinMEM Live Trading Mode — Real-Time Decision + Paper Execution

This is how you USE FinMEM as an application:

    python3 live_trade.py --ticker TSLA

What happens:
    1. TradingView → Gets real-time RSI, MACD, Bollinger Bands
    2. FinMEM Agent → Analyzes with layered memory + LLM
    3. Alpaca → Executes paper trade (free, no real money)

Setup (one-time):
    1. Alpaca: Sign up at https://app.alpaca.markets/signup
       → Paper Trading → API Keys → Generate
       → Add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env

    2. That's it. TradingView indicators need no API key.

Usage:
    python3 live_trade.py --ticker TSLA              # Single decision
    python3 live_trade.py --ticker TSLA --execute     # Decision + paper trade
    python3 live_trade.py --ticker TSLA --loop 60     # Every 60 seconds
    python3 live_trade.py --account                   # Check Alpaca account
    python3 live_trade.py --positions                 # Show open positions
    python3 live_trade.py --indicators TSLA           # Just show indicators
"""

import os
import sys
import argparse
import json
import time
import logging
from datetime import datetime, date

# Load .env
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_trade")


def show_indicators(ticker: str):
    """Show TradingView technical indicators."""
    from finmem.execution.tradingview_indicators import (
        get_quick_signal,
        format_for_agent,
    )

    print(f"\n{'═'*60}")
    print(f"  📊 TradingView Analysis: {ticker}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*60}\n")

    # Get signal
    sig = get_quick_signal(ticker)

    if sig.get("error"):
        print(f"  ❌ Error: {sig['error']}")
        return sig

    # Pretty print
    print(f"  Signal:     {sig['tv_recommendation']}")
    print(f"  Confidence: {sig['confidence']:.1%}")
    print(f"  Price:      ${sig['price']:.2f}" if sig['price'] else "  Price: N/A")
    print()
    print(f"  ── Key Indicators ──")
    print(f"  RSI(14):    {sig['rsi']:.1f}" if sig['rsi'] else "  RSI: N/A")
    print(f"  MACD:       {sig['macd']:.3f}" if sig['macd'] else "  MACD: N/A")
    print(f"  EMA20:      {sig['ema20']:.2f}" if sig['ema20'] else "  EMA20: N/A")
    print(f"  EMA50:      {sig['ema50']:.2f}" if sig['ema50'] else "  EMA50: N/A")
    if sig['bb_upper'] and sig['bb_lower']:
        print(f"  Bollinger:  [{sig['bb_lower']:.2f} — {sig['bb_upper']:.2f}]")
    print(f"  ADX:        {sig['adx']:.1f}" if sig['adx'] else "  ADX: N/A")
    print()
    print(f"  Indicators: {sig['buy_indicators']} BUY | "
          f"{sig['sell_indicators']} SELL | "
          f"{sig['neutral_indicators']} NEUTRAL")
    print(f"{'═'*60}\n")

    return sig


def make_decision(ticker: str) -> dict:
    """Run the full FinMEM + TradingView decision pipeline."""
    from finmem.execution.tradingview_indicators import get_quick_signal, format_for_agent
    from finmem.llm_client import LLMClient, ChatMessage
    from finmem.config import DEFAULT_CONFIG

    print(f"\n{'═'*60}")
    print(f"  🧠 FinMEM Live Decision: {ticker}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*60}")

    # Step 1: Get TradingView indicators
    print(f"\n  [1/3] Fetching TradingView indicators...")
    sig = get_quick_signal(ticker)

    if sig.get("error"):
        # Try NYSE if NASDAQ fails
        sig = get_quick_signal(ticker, exchange="NYSE")

    ta_text = format_for_agent(ticker)
    print(f"  ✅ Signal: {sig.get('tv_recommendation', '?')} | RSI: {sig.get('rsi', '?')}")

    # Step 2: LLM analysis
    print(f"  [2/3] Running LLM analysis (with memory context)...")
    llm = LLMClient(config=DEFAULT_CONFIG.llm)

    prompt = f"""You are a professional trading analyst using FinMEM's memory-augmented system.

Based on the following technical analysis, provide a trading decision.

{ta_text}

Current date: {datetime.now().strftime('%Y-%m-%d')}

Respond in this exact JSON format:
{{
    "decision": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "rationale": "Brief explanation (1-2 sentences)"
}}

Only output valid JSON, nothing else."""

    response = llm.chat(prompt)

    # Parse LLM response
    try:
        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"decision": "HOLD", "confidence": 0.3, "rationale": "Could not parse LLM response"}
    except Exception:
        result = {"decision": "HOLD", "confidence": 0.3, "rationale": "LLM parse error"}

    decision = result.get("decision", "HOLD").upper()
    confidence = float(result.get("confidence", 0.5))
    rationale = result.get("rationale", "N/A")

    print(f"  ✅ Decision: {decision} (confidence: {confidence:.1%})")
    print(f"  📝 Rationale: {rationale}")

    return {
        "ticker": ticker,
        "decision": decision,
        "confidence": confidence,
        "rationale": rationale,
        "tv_signal": sig.get("tv_recommendation"),
        "rsi": sig.get("rsi"),
        "price": sig.get("price"),
        "timestamp": datetime.now().isoformat(),
    }


def execute_trade(ticker: str, decision: str, confidence: float):
    """Execute on Alpaca paper trading."""
    from finmem.execution.alpaca_executor import execute_decision

    print(f"\n  [3/3] Executing on Alpaca Paper Trading...")

    result = execute_decision(
        ticker=ticker,
        decision=decision,
        confidence=confidence,
    )

    if result["status"] == "skipped":
        print(f"  ⏭️  HOLD — no order submitted")
    elif "FAILED" in str(result.get("status", "")):
        print(f"  ❌ Order failed: {result['status']}")
    else:
        print(f"  ✅ {result['action']} {result.get('shares', 0)} {ticker} "
              f"@ ~${result.get('estimated_price', 0):.2f}")
        print(f"  📋 Order ID: {result.get('order_id', 'N/A')}")

    return result


def show_account():
    """Show Alpaca account info."""
    from finmem.execution.alpaca_executor import get_account_info

    print(f"\n{'═'*60}")
    print(f"  💰 Alpaca Paper Trading Account")
    print(f"{'═'*60}")

    info = get_account_info()
    print(f"  Status:          {info['status']}")
    print(f"  Cash:            ${info['cash']:,.2f}")
    print(f"  Portfolio Value: ${info['portfolio_value']:,.2f}")
    print(f"  Buying Power:    ${info['buying_power']:,.2f}")
    print(f"  Equity:          ${info['equity']:,.2f}")
    print(f"{'═'*60}\n")


def show_positions():
    """Show open positions."""
    from finmem.execution.alpaca_executor import get_positions

    positions = get_positions()

    print(f"\n{'═'*60}")
    print(f"  📈 Open Positions ({len(positions)})")
    print(f"{'═'*60}")

    if not positions:
        print(f"  No open positions.")
    else:
        for p in positions:
            pl_pct = p['unrealized_plpc'] * 100
            emoji = "🟢" if pl_pct >= 0 else "🔴"
            print(f"  {emoji} {p['ticker']:6s} | {p['shares']:.0f} shares | "
                  f"${p['market_value']:,.2f} | P&L: {pl_pct:+.2f}%")

    print(f"{'═'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="FinMEM Live Trading — AI Decision + Paper Execution"
    )
    parser.add_argument("--ticker", "-t", type=str, default="TSLA",
                        help="Stock ticker (default: TSLA)")
    parser.add_argument("--execute", "-x", action="store_true",
                        help="Execute the decision on Alpaca paper trading")
    parser.add_argument("--loop", type=int, default=None,
                        help="Re-run every N seconds (e.g. --loop 60)")
    parser.add_argument("--indicators", "-i", type=str, default=None,
                        help="Just show TradingView indicators for a ticker")
    parser.add_argument("--account", action="store_true",
                        help="Show Alpaca account info")
    parser.add_argument("--positions", action="store_true",
                        help="Show open positions")
    parser.add_argument("--orders", action="store_true",
                        help="Show recent orders")

    args = parser.parse_args()

    # Info-only commands
    if args.account:
        show_account()
        return

    if args.positions:
        show_positions()
        return

    if args.orders:
        from finmem.execution.alpaca_executor import get_recent_orders
        orders = get_recent_orders()
        print(f"\n  Recent Orders ({len(orders)}):")
        for o in orders:
            print(f"  {o['side']:4s} {o['ticker']:6s} | "
                  f"qty: {o['qty']:.0f} | status: {o['status']} | "
                  f"{o['submitted_at'][:19]}")
        return

    if args.indicators:
        show_indicators(args.indicators)
        return

    # Main decision loop
    ticker = args.ticker

    while True:
        try:
            # Make decision
            result = make_decision(ticker)

            # Execute if flag set
            if args.execute:
                execute_trade(ticker, result["decision"], result["confidence"])

            print(f"\n{'─'*60}")
            print(f"  Summary: {result['decision']} {ticker} "
                  f"(conf: {result['confidence']:.1%}) "
                  f"| TV: {result['tv_signal']} "
                  f"| RSI: {result.get('rsi', 'N/A')}")
            print(f"{'─'*60}\n")

        except KeyboardInterrupt:
            print("\n\n  👋 Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error: {e}")

        # Loop or exit
        if args.loop:
            print(f"  ⏳ Next check in {args.loop} seconds... (Ctrl+C to stop)")
            try:
                time.sleep(args.loop)
            except KeyboardInterrupt:
                print("\n\n  👋 Stopped by user.")
                break
        else:
            break


if __name__ == "__main__":
    main()
