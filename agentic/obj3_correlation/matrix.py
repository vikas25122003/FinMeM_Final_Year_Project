"""
Objective 3 — Step 1: Rolling Pearson Correlation Matrix

Downloads price history for all portfolio tickers via yfinance,
computes daily returns, and calculates a 30-day rolling Pearson
correlation matrix. Result is cached per day.
"""

import os
import logging
from datetime import date, datetime
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Day-level cache to avoid repeated yfinance calls
_cache: dict = {"date": None, "matrix": None}


def compute_correlation_matrix(
    tickers: Optional[List[str]] = None,
    window: Optional[int] = None,
    period: str = "1y",
    reference_date: Optional[str] = None,
) -> pd.DataFrame:
    """Compute rolling Pearson correlation matrix for portfolio tickers.

    Args:
        tickers:        List of tickers. Defaults to PORTFOLIO_TICKERS env var.
        window:         Rolling window in days. Defaults to CORRELATION_WINDOW_DAYS.
        period:         yfinance download period (default: 1 year).
        reference_date: Date string (YYYY-MM-DD) for caching. Defaults to today.

    Returns:
        pd.DataFrame — tickers × tickers correlation matrix.
        Falls back to identity matrix on any error.
    """
    import yfinance as yf

    if tickers is None:
        tickers_str = os.getenv("PORTFOLIO_TICKERS", "TSLA,NVDA,MSFT,AMZN,NFLX")
        tickers = [t.strip() for t in tickers_str.split(",")]

    if window is None:
        window = int(os.getenv("CORRELATION_WINDOW_DAYS", "30"))

    cache_key = reference_date or str(date.today())

    # Check cache
    if _cache["date"] == cache_key and _cache["matrix"] is not None:
        logger.debug(f"[Obj3] Using cached correlation matrix for {cache_key}")
        return _cache["matrix"]

    try:
        logger.info(f"[Obj3] Downloading price data for {tickers}...")

        # Download all tickers at once
        data = yf.download(tickers, period=period, progress=False, auto_adjust=True)

        if data is None or data.empty:
            logger.warning("[Obj3] No price data downloaded — using identity matrix")
            return _identity_matrix(tickers)

        # Extract close prices
        if "Close" in data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else "Close" in data.columns:
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data["Close"]
            else:
                close_prices = data[["Close"]]
                close_prices.columns = tickers[:1]
        else:
            logger.warning("[Obj3] No 'Close' column in data")
            return _identity_matrix(tickers)

        # Handle single ticker case
        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame(name=tickers[0])

        # Compute daily returns
        daily_returns = close_prices.pct_change().dropna()

        if len(daily_returns) < window:
            logger.warning(
                f"[Obj3] Only {len(daily_returns)} days of data, "
                f"need at least {window} for rolling window. Using full-sample correlation."
            )
            corr_matrix = daily_returns.corr()
        else:
            # Compute rolling correlation on the last `window` days
            recent_returns = daily_returns.tail(window)
            corr_matrix = recent_returns.corr()

        # Ensure all tickers are present (some may be missing from yfinance)
        for t in tickers:
            if t not in corr_matrix.columns:
                corr_matrix[t] = 0.0
                corr_matrix.loc[t] = 0.0
                corr_matrix.loc[t, t] = 1.0

        # Reorder to match input ticker order
        corr_matrix = corr_matrix.reindex(index=tickers, columns=tickers, fill_value=0.0).copy()

        # Fill diagonal with 1.0 (use .copy() to avoid read-only issues)
        vals = corr_matrix.values.copy()
        np.fill_diagonal(vals, 1.0)
        corr_matrix = pd.DataFrame(vals, index=tickers, columns=tickers)

        # Fill any NaN with 0.0
        corr_matrix = corr_matrix.fillna(0.0)

        # Cache
        _cache["date"] = cache_key
        _cache["matrix"] = corr_matrix

        logger.info(f"[Obj3] Correlation matrix computed ({len(tickers)}×{len(tickers)})")
        return corr_matrix

    except Exception as exc:
        logger.warning(f"[Obj3] Correlation computation failed: {exc}")
        return _identity_matrix(tickers)


def _identity_matrix(tickers: List[str]) -> pd.DataFrame:
    """Create an identity correlation matrix (no correlation between tickers).

    Used as fallback when data download fails.
    """
    n = len(tickers)
    data = np.eye(n)
    return pd.DataFrame(data, index=tickers, columns=tickers)


def clear_cache() -> None:
    """Clear the cached correlation matrix."""
    _cache["date"] = None
    _cache["matrix"] = None
