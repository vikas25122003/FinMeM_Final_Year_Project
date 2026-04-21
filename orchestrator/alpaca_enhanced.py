"""
Enhanced Alpaca Executor — Full-featured paper trading with limit orders,
stop-losses, fill tracking, and market-hours awareness.

Wraps alpaca-py with everything needed for autonomous operation.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _get_trading_client():
    """Lazy-import and return Alpaca TradingClient (paper mode)."""
    from alpaca.trading.client import TradingClient

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")
    return TradingClient(api_key, secret_key, paper=True)


def _get_data_client():
    """Lazy-import and return Alpaca StockHistoricalDataClient."""
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
    )


def is_market_open() -> bool:
    """Check if the US stock market is currently open via Alpaca clock API."""
    try:
        client = _get_trading_client()
        clock = client.get_clock()
        return clock.is_open
    except Exception as e:
        logger.warning(f"[Alpaca] Could not check market status: {e}")
        return False


def get_next_open() -> Optional[datetime]:
    """Get the next market open time."""
    try:
        client = _get_trading_client()
        clock = client.get_clock()
        return clock.next_open
    except Exception:
        return None


def get_next_close() -> Optional[datetime]:
    """Get the next market close time."""
    try:
        client = _get_trading_client()
        clock = client.get_clock()
        return clock.next_close
    except Exception:
        return None


def get_latest_price(ticker: str) -> Optional[float]:
    """Get the latest trade price for a ticker."""
    try:
        from alpaca.data.requests import StockLatestTradeRequest

        data_client = _get_data_client()
        request = StockLatestTradeRequest(symbol_or_symbols=ticker)
        latest = data_client.get_stock_latest_trade(request)
        return float(latest[ticker].price)
    except Exception as e:
        logger.warning(f"[Alpaca] Could not get price for {ticker}: {e}")
        return None


def get_account_info() -> Dict[str, Any]:
    """Get paper account status and balance."""
    client = _get_trading_client()
    account = client.get_account()
    market_open = False
    try:
        clock = client.get_clock()
        market_open = clock.is_open
    except Exception:
        pass
    return {
        "status": account.status.value,
        "cash": float(account.cash),
        "portfolio_value": float(account.portfolio_value),
        "buying_power": float(account.buying_power),
        "equity": float(account.equity),
        "last_equity": float(account.last_equity),
        "market_open": market_open,
    }


def get_positions() -> Dict[str, Dict[str, Any]]:
    """Get all open positions as ticker -> info dict."""
    client = _get_trading_client()
    positions = client.get_all_positions()
    return {
        p.symbol: {
            "shares": float(p.qty),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
            "current_price": float(p.current_price),
            "avg_entry": float(p.avg_entry_price),
        }
        for p in positions
    }


def submit_limit_buy(
    ticker: str,
    qty: int,
    limit_price: float,
) -> Dict[str, Any]:
    """Submit a limit buy order with day time-in-force.

    Uses limit price = latest_price * 1.001 (0.1% buffer) if no
    explicit limit_price is given from caller.
    """
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = _get_trading_client()
    order_data = LimitOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=round(limit_price, 2),
    )

    try:
        order = client.submit_order(order_data)
        result = {
            "action": "BUY",
            "order_type": "limit",
            "ticker": ticker,
            "qty": qty,
            "limit_price": limit_price,
            "order_id": str(order.id),
            "status": order.status.value,
        }
        logger.info(f"[Alpaca] LIMIT BUY {qty} {ticker} @ ${limit_price:.2f} → {order.status.value}")
        return result
    except Exception as e:
        logger.error(f"[Alpaca] Limit buy failed: {e}")
        return {"action": "BUY", "ticker": ticker, "status": f"FAILED: {e}"}


def submit_market_sell(ticker: str, qty: int) -> Dict[str, Any]:
    """Submit a market sell order."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = _get_trading_client()

    try:
        position = client.get_open_position(ticker)
        available = int(float(position.qty))
        qty = min(qty, available)
        if qty <= 0:
            return {"action": "SELL", "ticker": ticker, "status": "no_position"}
    except Exception:
        return {"action": "SELL", "ticker": ticker, "status": "no_position"}

    order_data = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )

    try:
        order = client.submit_order(order_data)
        result = {
            "action": "SELL",
            "order_type": "market",
            "ticker": ticker,
            "qty": qty,
            "order_id": str(order.id),
            "status": order.status.value,
        }
        logger.info(f"[Alpaca] MARKET SELL {qty} {ticker} → {order.status.value}")
        return result
    except Exception as e:
        logger.error(f"[Alpaca] Market sell failed: {e}")
        return {"action": "SELL", "ticker": ticker, "status": f"FAILED: {e}"}


def submit_stop_loss(
    ticker: str,
    qty: int,
    stop_price: float,
) -> Dict[str, Any]:
    """Place a stop-loss order (GTC) to protect a position."""
    from alpaca.trading.requests import StopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = _get_trading_client()
    order_data = StopOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        stop_price=round(stop_price, 2),
    )

    try:
        order = client.submit_order(order_data)
        result = {
            "action": "STOP_LOSS",
            "ticker": ticker,
            "qty": qty,
            "stop_price": stop_price,
            "order_id": str(order.id),
            "status": order.status.value,
        }
        logger.info(f"[Alpaca] STOP LOSS {ticker} @ ${stop_price:.2f} for {qty} shares")
        return result
    except Exception as e:
        logger.error(f"[Alpaca] Stop loss failed: {e}")
        return {"action": "STOP_LOSS", "ticker": ticker, "status": f"FAILED: {e}"}


def wait_for_fill(order_id: str, timeout_seconds: int = 30) -> Dict[str, Any]:
    """Poll order status until filled or timeout.

    Returns order details including fill price if available.
    """
    client = _get_trading_client()
    start = time.time()
    interval = 1.0

    while time.time() - start < timeout_seconds:
        try:
            order = client.get_order_by_id(order_id)
            status = order.status.value

            if status == "filled":
                return {
                    "order_id": order_id,
                    "status": "filled",
                    "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                    "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0,
                    "filled_at": str(order.filled_at) if order.filled_at else None,
                }
            elif status in ("canceled", "expired", "rejected"):
                return {
                    "order_id": order_id,
                    "status": status,
                    "filled_qty": 0,
                }
        except Exception as e:
            logger.warning(f"[Alpaca] Order poll error: {e}")

        time.sleep(interval)
        interval = min(interval * 1.5, 5.0)

    return {"order_id": order_id, "status": "timeout", "filled_qty": 0}


def cancel_open_orders(ticker: Optional[str] = None) -> int:
    """Cancel all open orders, optionally filtered by ticker.

    Returns count of cancelled orders.
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    client = _get_trading_client()
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    orders = client.get_orders(request)

    cancelled = 0
    for order in orders:
        if ticker and order.symbol != ticker:
            continue
        try:
            client.cancel_order_by_id(order.id)
            cancelled += 1
        except Exception:
            pass

    return cancelled


def execute_decision(
    ticker: str,
    action: str,
    target_shares: int,
    stop_loss_pct: float = 0.05,
) -> List[Dict[str, Any]]:
    """Execute a full decision: place order, wait for fill, set stop-loss.

    Args:
        ticker: Stock symbol.
        action: BUY, SELL, HOLD, OVERWEIGHT, UNDERWEIGHT.
        target_shares: Number of shares to trade.
        stop_loss_pct: Stop-loss distance as fraction (0.05 = 5%).

    Returns:
        List of order results (may include primary order + stop-loss).
    """
    action = action.upper().strip()
    results = []

    if action in ("HOLD",):
        logger.info(f"[Alpaca] HOLD {ticker} — no action")
        return [{"action": "HOLD", "ticker": ticker, "status": "skipped"}]

    if target_shares <= 0:
        logger.info(f"[Alpaca] {action} {ticker} — 0 shares, skipping")
        return [{"action": action, "ticker": ticker, "status": "zero_shares"}]

    if action in ("BUY", "OVERWEIGHT"):
        price = get_latest_price(ticker)
        if price is None:
            return [{"action": "BUY", "ticker": ticker, "status": "no_price"}]

        limit_price = round(price * 1.002, 2)  # 0.2% buffer above market
        order_result = submit_limit_buy(ticker, target_shares, limit_price)
        results.append(order_result)

        if order_result.get("order_id"):
            fill = wait_for_fill(order_result["order_id"], timeout_seconds=30)
            order_result.update(fill)

            if fill.get("status") == "filled" and fill.get("filled_avg_price"):
                stop_price = round(fill["filled_avg_price"] * (1 - stop_loss_pct), 2)
                sl_result = submit_stop_loss(
                    ticker, int(fill.get("filled_qty", target_shares)), stop_price
                )
                results.append(sl_result)

    elif action in ("SELL", "UNDERWEIGHT"):
        order_result = submit_market_sell(ticker, target_shares)
        results.append(order_result)

        if order_result.get("order_id"):
            fill = wait_for_fill(order_result["order_id"], timeout_seconds=15)
            order_result.update(fill)

    return results
