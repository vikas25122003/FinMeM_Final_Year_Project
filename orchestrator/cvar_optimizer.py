"""
CVaR Portfolio Optimizer — Replaces Pearson-only concentration guard.

Computes Conditional Value-at-Risk (CVaR) optimal portfolio weights using
historical returns, then sizes positions accordingly. Falls back to
equal-weight if optimization fails.

CVaR at confidence α measures the expected loss in the worst (1-α)% of
scenarios — strictly more informative than pairwise Pearson correlation
for tail risk management.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def _fetch_returns(
    tickers: List[str],
    window_days: int = 60,
    period: str = "1y",
) -> pd.DataFrame:
    """Fetch daily returns for portfolio tickers."""
    import yfinance as yf

    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
        close.columns = tickers[:1]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    returns = close.pct_change().dropna()

    if len(returns) > window_days:
        returns = returns.tail(window_days)

    return returns


def compute_cvar(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Compute CVaR (Expected Shortfall) for a given weight vector.

    Args:
        weights: Portfolio weight vector (N,).
        returns: Historical returns matrix (T, N).
        confidence: VaR confidence level (e.g. 0.95 = worst 5%).

    Returns:
        CVaR value (positive = loss).
    """
    portfolio_returns = returns @ weights
    var_threshold = np.percentile(portfolio_returns, (1 - confidence) * 100)
    tail_returns = portfolio_returns[portfolio_returns <= var_threshold]

    if len(tail_returns) == 0:
        return -var_threshold

    return -np.mean(tail_returns)


def optimize_cvar_weights(
    tickers: List[str],
    returns_df: pd.DataFrame,
    confidence: float = 0.95,
    max_weight: float = 0.35,
) -> Dict[str, float]:
    """Compute CVaR-optimal portfolio weights.

    Minimizes CVaR subject to:
      - Weights sum to 1
      - Each weight in [0, max_weight]
      - Long-only (no shorting)

    Args:
        tickers: Ticker symbols.
        returns_df: DataFrame of daily returns (columns = tickers).
        confidence: CVaR confidence level.
        max_weight: Max allocation per ticker.

    Returns:
        Dict of ticker -> optimal weight.
    """
    available = [t for t in tickers if t in returns_df.columns]
    if len(available) < 2:
        return {t: 1.0 / len(tickers) for t in tickers}

    returns = returns_df[available].values
    n = len(available)

    initial_weights = np.ones(n) / n

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    bounds = [(0.0, max_weight) for _ in range(n)]

    try:
        result = minimize(
            compute_cvar,
            initial_weights,
            args=(returns, confidence),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if result.success:
            weights = result.x
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
        else:
            logger.warning(f"[CVaR] Optimization did not converge: {result.message}")
            weights = initial_weights

    except Exception as e:
        logger.warning(f"[CVaR] Optimization failed: {e}")
        weights = initial_weights

    weight_dict = {}
    for i, ticker in enumerate(available):
        weight_dict[ticker] = float(weights[i])

    for ticker in tickers:
        if ticker not in weight_dict:
            weight_dict[ticker] = 1.0 / len(tickers)

    return weight_dict


def apply_cvar_portfolio_optimization(
    decisions: Dict[str, Dict[str, Any]],
    portfolio_value: float,
    confidence: float = 0.95,
    max_weight: float = 0.35,
    window_days: int = 60,
    concentration_threshold: float = 0.80,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Full CVaR pipeline: optimize weights, apply concentration guard, size positions.

    Args:
        decisions: ticker -> {action, confidence, regime, ...}
        portfolio_value: Current total portfolio value.
        confidence: CVaR confidence level.
        max_weight: Max weight per ticker.
        window_days: Lookback for returns.
        concentration_threshold: Pearson correlation threshold for guard.

    Returns:
        (modified_decisions, cvar_weights)
    """
    tickers = list(decisions.keys())

    if len(tickers) == 0:
        return decisions, {}

    returns_df = _fetch_returns(tickers, window_days)

    if returns_df.empty:
        logger.warning("[CVaR] No return data — using equal weights")
        equal_w = 1.0 / max(len(tickers), 1)
        weights = {t: equal_w for t in tickers}
    else:
        weights = optimize_cvar_weights(tickers, returns_df, confidence, max_weight)

        corr_matrix = returns_df.corr() if not returns_df.empty else pd.DataFrame()
        if not corr_matrix.empty:
            _apply_concentration_guard(decisions, corr_matrix, concentration_threshold)

    for ticker, dec in decisions.items():
        w = weights.get(ticker, 0.1)
        action = dec.get("action", "HOLD").upper()
        conf = dec.get("confidence", 0.5)

        if action in ("BUY", "OVERWEIGHT"):
            dollar_amount = portfolio_value * w * conf
            price = dec.get("price", 0)
            if price and price > 0:
                dec["target_shares"] = max(1, int(dollar_amount / price))
            else:
                dec["target_shares"] = 0
            dec["cvar_weight"] = w
        elif action in ("SELL", "UNDERWEIGHT"):
            dec["target_shares"] = dec.get("current_shares", 0)
            dec["cvar_weight"] = w
        else:
            dec["target_shares"] = 0
            dec["cvar_weight"] = w

    logger.info(f"[CVaR] Weights: {weights}")
    return decisions, weights


def _apply_concentration_guard(
    decisions: Dict[str, Dict[str, Any]],
    corr_matrix: pd.DataFrame,
    threshold: float,
) -> int:
    """Override lower-confidence BUY to HOLD when pairwise correlation is too high."""
    from itertools import combinations

    buy_tickers = [
        t for t, d in decisions.items()
        if d.get("action", "").upper() in ("BUY", "OVERWEIGHT")
    ]

    if len(buy_tickers) < 2:
        return 0

    guard_count = 0
    overridden = set()

    for t1, t2 in combinations(buy_tickers, 2):
        if t1 in overridden or t2 in overridden:
            continue

        try:
            corr_val = abs(float(corr_matrix.loc[t1, t2]))
        except (KeyError, ValueError):
            continue

        if corr_val > threshold:
            conf1 = float(decisions[t1].get("confidence", 0.5))
            conf2 = float(decisions[t2].get("confidence", 0.5))

            override_ticker = t2 if conf1 >= conf2 else t1

            decisions[override_ticker]["action"] = "HOLD"
            decisions[override_ticker]["override_reason"] = (
                f"CVaR concentration guard: corr={corr_val:.2f} with "
                f"{t1 if override_ticker == t2 else t2}"
            )
            overridden.add(override_ticker)
            guard_count += 1
            logger.info(f"[CVaR] Guard: {override_ticker} BUY→HOLD (corr={corr_val:.2f})")

    return guard_count
