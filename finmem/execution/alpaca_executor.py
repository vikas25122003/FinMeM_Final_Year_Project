"""
Alpaca Paper Trading Executor — Connects FinMEM to Alpaca Markets.

Takes FinMEM agent decisions (BUY/SELL/HOLD) and submits real paper trades
to Alpaca's free paper trading account.

Setup:
    1. Sign up at https://app.alpaca.markets/signup (free)
    2. Go to Paper Trading → API Keys → Generate
    3. Add to your .env:
        ALPACA_API_KEY=your_key_here
        ALPACA_SECRET_KEY=your_secret_here

No real money involved. Ever.
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def _get_client():
    """Get Alpaca trading client (lazy import)."""
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        raise ImportError(
            "alpaca-py not installed. Run: pip install alpaca-py"
        )

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env. "
            "Sign up free at https://app.alpaca.markets/signup"
        )

    # paper=True → paper trading (no real money)
    return TradingClient(api_key, secret_key, paper=True)


def get_account_info() -> Dict[str, Any]:
    """Get Alpaca paper account status and balance."""
    client = _get_client()
    account = client.get_account()
    return {
        "status": account.status.value,
        "cash": float(account.cash),
        "portfolio_value": float(account.portfolio_value),
        "buying_power": float(account.buying_power),
        "equity": float(account.equity),
        "currency": account.currency,
    }


def execute_decision(
    ticker: str,
    decision: str,
    confidence: float = 0.5,
    portfolio_value: float = 100000.0,
    position_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Execute a FinMEM decision on Alpaca paper trading.

    Args:
        ticker: Stock symbol (e.g. "TSLA")
        decision: "BUY", "SELL", or "HOLD"
        confidence: Agent's confidence (0-1)
        portfolio_value: Total portfolio value for sizing
        position_pct: Max % of portfolio per trade (default 5%)

    Returns:
        Order result dict with status, shares, price, order_id.
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    decision = decision.upper().strip()

    if decision == "HOLD":
        logger.info(f"[Alpaca] HOLD {ticker} — no order")
        return {"action": "HOLD", "ticker": ticker, "status": "skipped"}

    client = _get_client()

    # Get current price from latest trade
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        data_client = StockHistoricalDataClient(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_SECRET_KEY"),
        )
        request = StockLatestTradeRequest(symbol_or_symbols=ticker)
        latest = data_client.get_stock_latest_trade(request)
        price = float(latest[ticker].price)
    except Exception as e:
        logger.warning(f"[Alpaca] Could not get price, using estimate: {e}")
        price = 200.0  # fallback

    # Position sizing: confidence-weighted % of portfolio
    max_dollars = portfolio_value * position_pct * confidence
    shares = max(1, int(max_dollars / price))

    side = OrderSide.BUY if decision == "BUY" else OrderSide.SELL

    # Check if we have shares to sell
    if side == OrderSide.SELL:
        try:
            position = client.get_open_position(ticker)
            available = int(position.qty)
            shares = min(shares, available)
            if shares <= 0:
                return {"action": "SELL", "ticker": ticker, "status": "no_position"}
        except Exception:
            return {"action": "SELL", "ticker": ticker, "status": "no_position"}

    try:
        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(order_data)

        result = {
            "action": decision,
            "ticker": ticker,
            "shares": shares,
            "estimated_price": price,
            "estimated_value": round(shares * price, 2),
            "order_id": str(order.id),
            "status": order.status.value,
            "submitted_at": str(order.submitted_at),
            "confidence": confidence,
        }
        logger.info(
            f"[Alpaca] {decision} {shares} {ticker} @ ~${price:.2f} "
            f"(conf: {confidence:.2f}) → {order.status.value}"
        )
        return result

    except Exception as e:
        logger.error(f"[Alpaca] Order failed: {e}")
        return {"action": decision, "ticker": ticker, "status": f"FAILED: {e}"}


def get_positions() -> list:
    """Get all current open positions."""
    client = _get_client()
    positions = client.get_all_positions()
    return [
        {
            "ticker": p.symbol,
            "shares": float(p.qty),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
            "current_price": float(p.current_price),
            "avg_entry": float(p.avg_entry_price),
        }
        for p in positions
    ]


def get_recent_orders(limit: int = 10) -> list:
    """Get recent orders."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    client = _get_client()
    request = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        limit=limit,
    )
    orders = client.get_orders(request)
    return [
        {
            "ticker": o.symbol,
            "side": o.side.value,
            "qty": float(o.qty) if o.qty else 0,
            "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
            "status": o.status.value,
            "submitted_at": str(o.submitted_at),
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
        }
        for o in orders
    ]
