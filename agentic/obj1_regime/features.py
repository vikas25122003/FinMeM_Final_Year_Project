"""
Objective 1 — Step 1: Feature Engineering for Regime Detection

Computes three daily market features from yfinance price data:
  - vol_20d:    20-day realized annualized volatility
  - vol_50d:    50-day realized annualized volatility
  - momentum_20d: 20-day price return

No FRED/VIX API key required. Uses price-derived volatility only.
Results are cached per (ticker, date_str) to avoid redundant API calls
within a single simulation run.

Usage:
    features = compute_features("TSLA", "2022-10-25")
    # {'vol_20d': 0.74, 'vol_50d': 0.65, 'momentum_20d': -0.15}
"""

import logging
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Safe defaults returned when data is unavailable (SIDEWAYS regime values)
_DEFAULTS: Dict[str, float] = {
    "vol_20d": 0.20,
    "vol_50d": 0.20,
    "momentum_20d": 0.0,
}


@lru_cache(maxsize=512)
def compute_features(ticker: str, date_str: str, lookback_days: int = 90) -> Dict[str, float]:
    """
    Compute regime detection features for a given ticker and date.

    Args:
        ticker:        Stock ticker symbol (e.g. "TSLA").
        date_str:      Date string in YYYY-MM-DD format.
        lookback_days: How many calendar days of history to fetch (default 90).

    Returns:
        Dict with keys: vol_20d, vol_50d, momentum_20d.
        All values are floats. Falls back to safe defaults on any error.
    """
    try:
        end_dt   = pd.Timestamp(date_str)
        start_dt = end_dt - pd.Timedelta(days=lookback_days + 10)  # buffer

        df = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        # Handle multi-level column headers (yfinance ≥0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty or len(df) < 22:
            logger.debug(
                f"[Obj1] Insufficient data for {ticker} on {date_str} "
                f"(got {len(df)} rows). Using defaults."
            )
            return dict(_DEFAULTS)

        close = df["Close"].squeeze()
        returns = close.pct_change().dropna()

        if len(returns) < 20:
            logger.debug(f"[Obj1] Too few return rows for {ticker} on {date_str}. Using defaults.")
            return dict(_DEFAULTS)

        # ── Annualised volatility (std of daily returns × √252) ──────────────
        vol_20d = float(returns.rolling(20).std().iloc[-1]) * (252 ** 0.5)
        vol_50d_window = min(50, len(returns))
        vol_50d = float(returns.rolling(vol_50d_window).std().iloc[-1]) * (252 ** 0.5)

        # ── 20-day momentum (price return over last 20 trading days) ──────────
        if len(close) >= 21:
            momentum_20d = float((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21])
        else:
            momentum_20d = 0.0

        # Guard NaNs
        if np.isnan(vol_20d):
            vol_20d = _DEFAULTS["vol_20d"]
        if np.isnan(vol_50d):
            vol_50d = _DEFAULTS["vol_50d"]
        if np.isnan(momentum_20d):
            momentum_20d = _DEFAULTS["momentum_20d"]

        features = {
            "vol_20d":      round(vol_20d, 6),
            "vol_50d":      round(vol_50d, 6),
            "momentum_20d": round(momentum_20d, 6),
        }

        logger.debug(f"[Obj1] Features for {ticker} on {date_str}: {features}")
        return features

    except Exception as exc:
        logger.warning(
            f"[Obj1] compute_features failed for {ticker} on {date_str}: {exc}. "
            f"Returning defaults."
        )
        return dict(_DEFAULTS)
