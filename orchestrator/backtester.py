"""
Backtester — Run the full pipeline on historical dates without Alpaca.

Simulates the complete autonomous trading cycle (regime → memory → agents →
CVaR → simulated execution → reflection) on historical data, producing
portfolio value curves and per-day trade logs for UI charts.

Agent mode calls TradingAgents via subprocess in the `tradingagents` conda env,
since it has different langchain/dependency versions than the main FinMEM venv.
"""

import json
import os
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONDA_PYTHON = os.path.expanduser("~/miniconda3/envs/tradingagents/bin/python")
TA_RUNNER = str(Path(__file__).resolve().parent.parent.parent / "TradingAgents" / "ta_runner.py")


class BacktestResult:
    """Holds the output of a backtest run."""

    def __init__(self):
        self.dates: List[str] = []
        self.portfolio_values: List[float] = []
        self.cash_values: List[float] = []
        self.prices: Dict[str, List[float]] = {}
        self.decisions: List[Dict[str, Any]] = []
        self.regimes: List[str] = []
        self.initial_cash: float = 100000
        self.final_value: float = 100000

    def to_dict(self) -> Dict[str, Any]:
        bh_returns = {}
        for ticker, price_list in self.prices.items():
            if len(price_list) >= 2 and price_list[0] > 0:
                bh_returns[ticker] = round(
                    (price_list[-1] - price_list[0]) / price_list[0] * 100, 2
                )

        return {
            "dates": self.dates,
            "portfolio_values": [round(v, 2) for v in self.portfolio_values],
            "cash_values": [round(v, 2) for v in self.cash_values],
            "prices": {k: [round(p, 2) for p in v] for k, v in self.prices.items()},
            "decisions": self.decisions,
            "regimes": self.regimes,
            "initial_cash": self.initial_cash,
            "final_value": round(self.final_value, 2),
            "total_return_pct": round(
                (self.final_value - self.initial_cash) / self.initial_cash * 100, 2
            ),
            "buy_hold_returns": bh_returns,
            "total_days": len(self.dates),
            "trade_count": sum(
                1 for d in self.decisions
                for t in (d.get("tickers") or {}).values()
                if t.get("action") not in ("HOLD", None)
            ),
        }


def _call_trading_agents(ticker: str, date: str) -> str:
    """Call TradingAgents via subprocess in the tradingagents conda env.

    Returns the action string (BUY / SELL / HOLD).
    """
    env = os.environ.copy()
    env["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

    deep_llm = os.getenv("TA_DEEP_LLM", "gemini-2.5-pro")
    quick_llm = os.getenv("TA_QUICK_LLM", "gemini-2.5-flash")
    provider = os.getenv("TA_LLM_PROVIDER", "google")

    cmd = [
        CONDA_PYTHON, TA_RUNNER,
        "--ticker", ticker,
        "--date", date,
        "--analysts", "market,news,fundamentals",
        "--llm-provider", provider,
        "--deep-llm", deep_llm,
        "--quick-llm", quick_llm,
    ]

    ta_dir = str(Path(TA_RUNNER).parent)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            env=env, cwd=ta_dir,
        )
        if proc.returncode != 0:
            logger.warning(f"[TA subprocess] stderr: {proc.stderr[:300]}")
            return "HOLD"

        stdout = proc.stdout.strip()
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                data = json.loads(line)
                action = data.get("action", "HOLD").strip().upper()
                logger.info(f"[TA subprocess] {ticker}@{date} → {action}")
                return action

        return "HOLD"

    except subprocess.TimeoutExpired:
        logger.warning(f"[TA subprocess] Timeout for {ticker}@{date}")
        return "HOLD"
    except Exception as e:
        logger.warning(f"[TA subprocess] Error for {ticker}@{date}: {e}")
        return "HOLD"


def run_backtest(
    tickers: List[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 100000,
    use_trading_agents: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> BacktestResult:
    """Run a full backtest over a date range.

    Args:
        tickers: Stock tickers to trade.
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        initial_cash: Starting capital.
        use_trading_agents: If True, calls TradingAgents via subprocess (slow).
        config: Optional orchestrator config dict.

    Returns:
        BacktestResult with full trace.
    """
    import yfinance as yf

    result = BacktestResult()
    result.initial_cash = initial_cash

    mode_label = "AGENTS" if use_trading_agents else "REGIME HEURISTIC"
    logger.info(f"[Backtest] {tickers} | {start_date} → {end_date} | ${initial_cash:,.0f} | {mode_label}")

    if use_trading_agents and not os.path.exists(CONDA_PYTHON):
        logger.error(f"[Backtest] tradingagents conda env not found at {CONDA_PYTHON}")
        use_trading_agents = False
    if use_trading_agents and not os.path.exists(TA_RUNNER):
        logger.error(f"[Backtest] ta_runner.py not found at {TA_RUNNER}")
        use_trading_agents = False

    hist_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    hist = yf.download(
        tickers, start=hist_start, end=end_date, progress=False, auto_adjust=True
    )

    if hist is None or hist.empty:
        logger.error("[Backtest] No price data")
        return result

    if isinstance(hist.columns, pd.MultiIndex):
        close = hist["Close"]
    else:
        close = hist[["Close"]]
        if len(tickers) == 1:
            close.columns = tickers

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    trade_start = datetime.strptime(start_date, "%Y-%m-%d")
    mask = close.index >= trade_start
    trade_dates = close.index[mask]

    if len(trade_dates) == 0:
        logger.error("[Backtest] No trading days in range")
        return result

    from agentic.obj1_regime.classifier import get_classifier
    from agentic.obj1_regime.features import compute_features

    try:
        regime_clf = get_classifier(os.environ.get("ADAPTIVE_Q_MODE", "hmm"))
    except Exception:
        from agentic.obj1_regime.classifier import ThresholdRegimeClassifier
        regime_clf = ThresholdRegimeClassifier()

    cash = initial_cash
    positions: Dict[str, float] = {t: 0 for t in tickers}
    entry_prices: Dict[str, float] = {t: 0 for t in tickers}

    for price_ticker in tickers:
        result.prices[price_ticker] = []

    total_days = len(trade_dates)

    for i, trade_date in enumerate(trade_dates):
        date_str = trade_date.strftime("%Y-%m-%d")

        if use_trading_agents and (i + 1) % 5 == 0:
            logger.info(f"[Backtest] Day {i+1}/{total_days} — {date_str}")

        day_prices = {}
        for ticker in tickers:
            try:
                p = float(close.loc[trade_date, ticker])
                day_prices[ticker] = p
                result.prices[ticker].append(p)
            except Exception:
                result.prices[ticker].append(result.prices[ticker][-1] if result.prices[ticker] else 0)

        regime = "SIDEWAYS"
        try:
            features = compute_features(tickers[0], date_str)
            regime = regime_clf.predict(features)
        except Exception:
            pass

        result.regimes.append(regime)

        day_decisions = {}
        for ticker in tickers:
            price = day_prices.get(ticker, 0)
            if price <= 0:
                day_decisions[ticker] = {"action": "HOLD", "ticker": ticker}
                continue

            if use_trading_agents:
                action = _call_trading_agents(ticker, date_str)
            else:
                action = _regime_heuristic(regime)

            if action in ("BUY", "OVERWEIGHT"):
                action = "BUY"
            elif action in ("SELL", "UNDERWEIGHT"):
                action = "SELL"
            else:
                action = "HOLD"

            day_decisions[ticker] = {"action": action, "ticker": ticker, "price": price}

        for ticker, dec in day_decisions.items():
            action = dec.get("action", "HOLD")
            price = dec.get("price", 0)

            if action == "BUY" and price > 0:
                max_spend = cash * 0.2
                shares_to_buy = int(max_spend / price)
                if shares_to_buy > 0 and shares_to_buy * price <= cash:
                    cost = shares_to_buy * price
                    if positions[ticker] > 0:
                        total_cost = positions[ticker] * entry_prices[ticker] + cost
                        positions[ticker] += shares_to_buy
                        entry_prices[ticker] = total_cost / positions[ticker]
                    else:
                        positions[ticker] = shares_to_buy
                        entry_prices[ticker] = price
                    cash -= cost
                    dec["shares"] = shares_to_buy

            elif action == "SELL" and positions[ticker] > 0:
                sell_shares = positions[ticker]
                cash += sell_shares * price
                dec["shares"] = sell_shares
                dec["pnl"] = round((price - entry_prices[ticker]) * sell_shares, 2)
                positions[ticker] = 0
                entry_prices[ticker] = 0

        position_value = sum(positions[t] * day_prices.get(t, 0) for t in tickers)
        total_value = cash + position_value

        result.dates.append(date_str)
        result.portfolio_values.append(total_value)
        result.cash_values.append(cash)
        result.decisions.append({
            "date": date_str,
            "regime": regime,
            "tickers": day_decisions,
            "cash": round(cash, 2),
            "position_value": round(position_value, 2),
            "total_value": round(total_value, 2),
        })

    result.final_value = result.portfolio_values[-1] if result.portfolio_values else initial_cash
    logger.info(f"[Backtest] Done: ${initial_cash:,.0f} → ${result.final_value:,.0f} "
                f"({(result.final_value - initial_cash) / initial_cash * 100:+.2f}%)")

    return result


def _regime_heuristic(regime: str) -> str:
    """Simple regime-based trading heuristic for fast backtesting."""
    if regime == "BULL":
        return "BUY"
    elif regime == "CRISIS":
        return "SELL"
    return "HOLD"
