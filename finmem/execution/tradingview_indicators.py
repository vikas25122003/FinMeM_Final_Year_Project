"""
TradingView Technical Indicators — Uses tradingview-mcp-server under the hood.

Provides RSI, MACD, Bollinger Bands, EMAs, and AI-powered stock decisions
via the TradingView MCP server's Python API. All data sourced from Yahoo
Finance (100% free, no API key needed).

This module wraps the tradingview_mcp library for direct programmatic use
inside the FinMEM pipeline (no MCP server process needed).
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def get_technical_analysis(
    symbol: str,
    exchange: str = "NASDAQ",
    screener: str = "america",
    interval: str = "1d",
) -> Dict[str, Any]:
    """
    Get full TradingView technical analysis for a symbol.

    Args:
        symbol: Ticker symbol (e.g. "TSLA", "AAPL")
        exchange: Exchange name (e.g. "NASDAQ", "NYSE")
        screener: Market screener (e.g. "america", "crypto")
        interval: Timeframe ("1m", "5m", "15m", "1h", "4h", "1d", "1W", "1M")

    Returns:
        Dict with summary, oscillators, moving_averages, and raw indicators.
    """
    try:
        from tradingview_ta import TA_Handler, Interval
    except ImportError:
        raise ImportError("tradingview-ta not installed. Run: pip install tradingview-ta")

    interval_map = {
        "1m": Interval.INTERVAL_1_MINUTE,
        "5m": Interval.INTERVAL_5_MINUTES,
        "15m": Interval.INTERVAL_15_MINUTES,
        "1h": Interval.INTERVAL_1_HOUR,
        "4h": Interval.INTERVAL_4_HOURS,
        "1d": Interval.INTERVAL_1_DAY,
        "1W": Interval.INTERVAL_1_WEEK,
        "1M": Interval.INTERVAL_1_MONTH,
    }

    handler = TA_Handler(
        symbol=symbol,
        exchange=exchange,
        screener=screener,
        interval=interval_map.get(interval, Interval.INTERVAL_1_DAY),
    )

    analysis = handler.get_analysis()

    return {
        "symbol": symbol,
        "exchange": exchange,
        "interval": interval,
        "summary": {
            "recommendation": analysis.summary["RECOMMENDATION"],
            "buy": analysis.summary["BUY"],
            "sell": analysis.summary["SELL"],
            "neutral": analysis.summary["NEUTRAL"],
        },
        "oscillators": {
            "recommendation": analysis.oscillators["RECOMMENDATION"],
            "buy": analysis.oscillators["BUY"],
            "sell": analysis.oscillators["SELL"],
            "neutral": analysis.oscillators["NEUTRAL"],
        },
        "moving_averages": {
            "recommendation": analysis.moving_averages["RECOMMENDATION"],
            "buy": analysis.moving_averages["BUY"],
            "sell": analysis.moving_averages["SELL"],
            "neutral": analysis.moving_averages["NEUTRAL"],
        },
        "indicators": {
            "RSI": analysis.indicators.get("RSI"),
            "RSI[1]": analysis.indicators.get("RSI[1]"),
            "MACD.macd": analysis.indicators.get("MACD.macd"),
            "MACD.signal": analysis.indicators.get("MACD.signal"),
            "BB.upper": analysis.indicators.get("BB.upper"),
            "BB.lower": analysis.indicators.get("BB.lower"),
            "EMA20": analysis.indicators.get("EMA20"),
            "EMA50": analysis.indicators.get("EMA50"),
            "EMA200": analysis.indicators.get("EMA200"),
            "SMA20": analysis.indicators.get("SMA20"),
            "SMA50": analysis.indicators.get("SMA50"),
            "SMA200": analysis.indicators.get("SMA200"),
            "ADX": analysis.indicators.get("ADX"),
            "ATR": analysis.indicators.get("ATR"),
            "Stoch.K": analysis.indicators.get("Stoch.K"),
            "Stoch.D": analysis.indicators.get("Stoch.D"),
            "CCI20": analysis.indicators.get("CCI20"),
            "volume": analysis.indicators.get("volume"),
            "close": analysis.indicators.get("close"),
            "open": analysis.indicators.get("open"),
            "high": analysis.indicators.get("high"),
            "low": analysis.indicators.get("low"),
        },
    }


def get_quick_signal(symbol: str, exchange: str = "NASDAQ") -> Dict[str, Any]:
    """
    Get a quick BUY/SELL/HOLD signal with key indicators.

    Returns a simplified dict suitable for the Obj4 technical_node.
    """
    try:
        ta = get_technical_analysis(symbol, exchange)
    except Exception as e:
        logger.warning(f"[TradingView] Failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "rsi": None,
            "macd": None,
            "error": str(e),
        }

    rec = ta["summary"]["recommendation"]
    buy_count = ta["summary"]["buy"]
    sell_count = ta["summary"]["sell"]
    total = buy_count + sell_count + ta["summary"]["neutral"]

    # Map TradingView recommendation to our signal
    signal_map = {
        "STRONG_BUY": "BUY",
        "BUY": "BUY",
        "NEUTRAL": "HOLD",
        "SELL": "SELL",
        "STRONG_SELL": "SELL",
    }
    signal = signal_map.get(rec, "HOLD")

    # Confidence = proportion of indicators agreeing
    if signal == "BUY":
        confidence = buy_count / total if total > 0 else 0.5
    elif signal == "SELL":
        confidence = sell_count / total if total > 0 else 0.5
    else:
        confidence = 0.5

    return {
        "symbol": symbol,
        "signal": signal,
        "tv_recommendation": rec,
        "confidence": round(confidence, 3),
        "rsi": ta["indicators"]["RSI"],
        "macd": ta["indicators"]["MACD.macd"],
        "macd_signal": ta["indicators"]["MACD.signal"],
        "ema20": ta["indicators"]["EMA20"],
        "ema50": ta["indicators"]["EMA50"],
        "bb_upper": ta["indicators"]["BB.upper"],
        "bb_lower": ta["indicators"]["BB.lower"],
        "adx": ta["indicators"]["ADX"],
        "price": ta["indicators"]["close"],
        "buy_indicators": buy_count,
        "sell_indicators": sell_count,
        "neutral_indicators": ta["summary"]["neutral"],
    }


def format_for_agent(symbol: str, exchange: str = "NASDAQ") -> str:
    """
    Format technical analysis as a text summary for the LLM agent.

    This is designed to be injected into the Obj4 technical_node prompt.
    """
    sig = get_quick_signal(symbol, exchange)

    if sig.get("error"):
        return f"[TradingView] No data for {symbol}: {sig['error']}"

    lines = [
        f"=== TradingView Technical Analysis: {symbol} ===",
        f"Signal: {sig['tv_recommendation']} (confidence: {sig['confidence']:.1%})",
        f"Price: ${sig['price']:.2f}" if sig['price'] else "Price: N/A",
        f"RSI(14): {sig['rsi']:.1f}" if sig['rsi'] else "RSI: N/A",
        f"MACD: {sig['macd']:.3f} (signal: {sig['macd_signal']:.3f})"
        if sig['macd'] and sig['macd_signal'] else "MACD: N/A",
        f"EMA20: {sig['ema20']:.2f} | EMA50: {sig['ema50']:.2f}"
        if sig['ema20'] and sig['ema50'] else "EMAs: N/A",
        f"Bollinger: [{sig['bb_lower']:.2f} — {sig['bb_upper']:.2f}]"
        if sig['bb_lower'] and sig['bb_upper'] else "BB: N/A",
        f"ADX: {sig['adx']:.1f}" if sig['adx'] else "ADX: N/A",
        f"Indicators: {sig['buy_indicators']} BUY | {sig['sell_indicators']} SELL | {sig['neutral_indicators']} NEUTRAL",
    ]
    return "\n".join(lines)
