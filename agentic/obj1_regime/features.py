"""
Objective 1 — Step 1: Feature Engineering for Regime Detection

Computes daily market features for BOTH classifiers:
  For ThresholdClassifier (scalar features):
    - vol_20d:      20-day realized annualized volatility (std × √252)
    - vol_50d:      50-day realized annualized volatility
    - momentum_20d: 20-day price return

  For HMMClassifier (sequence feature):
    - returns_seq:  tuple of last 60 daily log-returns (hashable for LRU cache)
                    HMM needs the full sequence to predict the current hidden state,
                    not just summary statistics.

Results are LRU-cached per (ticker, date_str) — one yfinance call per day
across the entire simulation run.

Usage:
    features = compute_features("TSLA", "2022-10-25")
    # vol_20d=0.699, momentum_20d=-0.214, returns_seq=(0.01, -0.02, ...)
"""

import logging
from functools import lru_cache
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Safe defaults — yields SIDEWAYS regime from ThresholdClassifier
_DEFAULTS: Dict[str, Any] = {
    "vol_20d":      0.20,
    "vol_50d":      0.20,
    "momentum_20d": 0.0,
    "returns_seq":  tuple([0.0] * 60),   # 60 zeros → HMM sees flat market
}


@lru_cache(maxsize=512)
def compute_features(ticker: str, date_str: str, lookback_days: int = 120) -> Dict[str, Any]:
    """
    Compute regime detection features for a given ticker and date.

    Args:
        ticker:        Stock ticker symbol (e.g. "TSLA").
        date_str:      Date string in YYYY-MM-DD format.
        lookback_days: Calendar days of history to fetch (default 120 — enough for 60 trading days).

    Returns:
        Dict with keys: vol_20d (float), vol_50d (float),
                        momentum_20d (float), returns_seq (tuple[float]).
        Falls back to safe defaults on any error.
    """
    try:
        end_dt   = pd.Timestamp(date_str)
        start_dt = end_dt - pd.Timedelta(days=lookback_days + 10)   # small buffer

        df = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        # Handle multi-level column headers (yfinance ≥ 0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty or len(df) < 5:
            logger.debug(f"[Obj1] Too little data for {ticker} on {date_str}. Using defaults.")
            return dict(_DEFAULTS)

        close   = df["Close"].squeeze()
        returns = close.pct_change().dropna()

        if len(returns) < 5:
            return dict(_DEFAULTS)

        # ── Scalar features for ThresholdClassifier ──────────────────────────
        vol_20d = (
            float(returns.rolling(20).std().iloc[-1]) * (252 ** 0.5)
            if len(returns) >= 20 else _DEFAULTS["vol_20d"]
        )
        vol_50d = (
            float(returns.rolling(50).std().iloc[-1]) * (252 ** 0.5)
            if len(returns) >= 50 else vol_20d
        )
        momentum_20d = (
            float((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21])
            if len(close) >= 21 else 0.0
        )

        # Guard NaNs
        if np.isnan(vol_20d):      vol_20d = _DEFAULTS["vol_20d"]
        if np.isnan(vol_50d):      vol_50d = _DEFAULTS["vol_50d"]
        if np.isnan(momentum_20d): momentum_20d = _DEFAULTS["momentum_20d"]

        # ── Sequence feature for HMMClassifier ───────────────────────────────
        # We use log-returns (~= pct_change for small moves) for better normality
        log_returns  = np.log(close / close.shift(1)).dropna()
        seq_values   = log_returns.values[-60:]

        if len(seq_values) < 60:
            # Left-pad with zeros if we have fewer than 60 days of history
            seq_values = np.concatenate([
                np.zeros(60 - len(seq_values)),
                seq_values
            ])

        # Replace any NaNs/Infs with 0 (safety)
        seq_values   = np.where(np.isfinite(seq_values), seq_values, 0.0)
        returns_seq  = tuple(float(x) for x in seq_values)

        features = {
            "vol_20d":      round(vol_20d, 6),
            "vol_50d":      round(vol_50d, 6),
            "momentum_20d": round(momentum_20d, 6),
            "returns_seq":  returns_seq,              # 60-element tuple
        }

        logger.debug(
            f"[Obj1] Features for {ticker} on {date_str}: "
            f"vol={vol_20d:.4f}, mom={momentum_20d:.4f}"
        )
        return features

    except Exception as exc:
        logger.warning(
            f"[Obj1] compute_features failed for {ticker} on {date_str}: {exc}. "
            "Using defaults."
        )
        return dict(_DEFAULTS)
