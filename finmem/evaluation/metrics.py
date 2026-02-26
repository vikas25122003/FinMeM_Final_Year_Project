"""
Evaluation Metrics — Paper-Faithful

Computes the 5 metrics reported in the FinMEM paper:
1. Cumulative Return (CR)
2. Sharpe Ratio
3. Annualized Volatility
4. Daily Volatility
5. Max Drawdown
"""

import numpy as np
from typing import List, Dict, Any, Optional


def compute_metrics(
    portfolio_values: List[float],
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> Dict[str, Any]:
    """Compute the paper's 5 evaluation metrics from portfolio value series.
    
    Args:
        portfolio_values: List of daily portfolio total values.
        risk_free_rate: Annual risk-free rate (default 0).
        trading_days_per_year: Trading days per year (default 252).
        
    Returns:
        Dictionary with all 5 metrics.
    """
    if len(portfolio_values) < 2:
        return {
            "cumulative_return": 0.0,
            "cumulative_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "annualized_volatility": 0.0,
            "daily_volatility": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "total_days": len(portfolio_values),
        }
    
    values = np.array(portfolio_values, dtype=float)
    initial = values[0]
    final = values[-1]
    
    # 1. Cumulative Return
    cumulative_return = final - initial
    cumulative_return_pct = (cumulative_return / initial) * 100 if initial > 0 else 0.0
    
    # Daily returns
    daily_returns = np.diff(values) / values[:-1]
    
    # 2. Daily Volatility
    daily_volatility = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    
    # 3. Annualized Volatility
    annualized_volatility = daily_volatility * np.sqrt(trading_days_per_year)
    
    # 4. Sharpe Ratio
    daily_rf = risk_free_rate / trading_days_per_year
    excess_returns = daily_returns - daily_rf
    mean_excess = float(np.mean(excess_returns))
    std_excess = float(np.std(excess_returns, ddof=1)) if len(excess_returns) > 1 else 1e-9
    sharpe_ratio = (mean_excess / std_excess) * np.sqrt(trading_days_per_year) if std_excess > 1e-9 else 0.0
    
    # 5. Max Drawdown
    cumulative = values / initial  # Normalize to start at 1.0
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_drawdown_pct = float(np.max(drawdowns)) * 100  # As percentage
    max_drawdown = float(np.max(running_max * initial - cumulative * initial))  # Dollar amount
    
    return {
        "cumulative_return": round(cumulative_return, 2),
        "cumulative_return_pct": round(cumulative_return_pct, 2),
        "sharpe_ratio": round(float(sharpe_ratio), 4),
        "annualized_volatility": round(float(annualized_volatility), 4),
        "daily_volatility": round(float(daily_volatility), 6),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "total_days": len(portfolio_values),
    }


def compute_buy_and_hold(
    prices: List[float],
    initial_capital: float = 100000.0,
) -> Dict[str, Any]:
    """Compute Buy & Hold baseline metrics for comparison.
    
    Paper: B&H is the primary baseline.
    
    Args:
        prices: Daily closing prices.
        initial_capital: Starting capital.
        
    Returns:
        Metrics dict for buy-and-hold strategy.
    """
    if len(prices) < 2:
        return compute_metrics([initial_capital])
    
    # Buy as many shares as possible at first price
    shares = initial_capital / prices[0]
    portfolio_values = [shares * p for p in prices]
    
    return compute_metrics(portfolio_values)


def format_metrics_report(
    agent_metrics: Dict[str, Any],
    bh_metrics: Optional[Dict[str, Any]] = None,
    ticker: str = "",
) -> str:
    """Format metrics into a readable report string.
    
    Args:
        agent_metrics: FinMEM agent metrics.
        bh_metrics: Buy & Hold baseline metrics (optional).
        ticker: Stock ticker.
        
    Returns:
        Formatted report string.
    """
    lines = []
    lines.append(f"  {'Metric':<25} {'FinMEM':>12}")
    if bh_metrics:
        lines.append(f"{'B&H':>12}")
    lines.append("")
    
    metrics_display = [
        ("Cumulative Return", "cumulative_return_pct", "%"),
        ("Sharpe Ratio", "sharpe_ratio", ""),
        ("Annualized Volatility", "annualized_volatility", ""),
        ("Daily Volatility", "daily_volatility", ""),
        ("Max Drawdown", "max_drawdown_pct", "%"),
    ]
    
    for label, key, suffix in metrics_display:
        agent_val = agent_metrics.get(key, 0)
        line = f"  {label:<25} {agent_val:>10.4f}{suffix}"
        if bh_metrics:
            bh_val = bh_metrics.get(key, 0)
            line += f"  {bh_val:>10.4f}{suffix}"
        lines.append(line)
    
    return "\n".join(lines)
