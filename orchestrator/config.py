"""
Unified configuration for the autonomous trading system.

Centralizes all settings: tickers, LLM providers, Alpaca keys,
scheduling, and feature flags.
"""

import os
from typing import Dict, Any, List


def get_orchestrator_config() -> Dict[str, Any]:
    """Build the unified config from environment variables with sane defaults."""

    portfolio_tickers_str = os.getenv("PORTFOLIO_TICKERS", "TSLA,NVDA,MSFT,AMZN,AAPL")
    portfolio_tickers = [t.strip() for t in portfolio_tickers_str.split(",")]

    return {
        # ── Portfolio ────────────────────────────────────────
        "portfolio_tickers": portfolio_tickers,
        "initial_cash": float(os.getenv("INITIAL_CASH", "100000")),

        # ── Scheduling ───────────────────────────────────────
        "cycle_interval_seconds": int(os.getenv("CYCLE_INTERVAL", "1800")),  # 30 min
        "market_open_hour": 9,
        "market_open_minute": 30,
        "market_close_hour": 16,
        "market_close_minute": 0,
        "timezone": "US/Eastern",

        # ── FinMEM Features ──────────────────────────────────
        "adaptive_q": os.getenv("ADAPTIVE_Q", "true").lower() == "true",
        "adaptive_q_mode": os.getenv("ADAPTIVE_Q_MODE", "hmm"),
        "learned_importance": os.getenv("LEARNED_IMPORTANCE", "true").lower() == "true",
        "cross_ticker": os.getenv("CROSS_TICKER", "true").lower() == "true",

        # ── CVaR Optimizer ───────────────────────────────────
        "cvar_confidence": float(os.getenv("CVAR_CONFIDENCE", "0.95")),
        "cvar_max_weight": float(os.getenv("CVAR_MAX_WEIGHT", "0.35")),
        "concentration_threshold": float(os.getenv("CONCENTRATION_THRESHOLD", "0.80")),
        "correlation_window_days": int(os.getenv("CORRELATION_WINDOW_DAYS", "60")),

        # ── Alpaca ───────────────────────────────────────────
        "alpaca_api_key": os.getenv("ALPACA_API_KEY", ""),
        "alpaca_secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
        "alpaca_paper": True,
        "stop_loss_pct": float(os.getenv("STOP_LOSS_PCT", "0.05")),
        "max_position_pct": float(os.getenv("MAX_POSITION_PCT", "0.25")),

        # ── TradingAgents LLM (Gemini) ──────────────────────
        "ta_llm_provider": os.getenv("TA_LLM_PROVIDER", "google"),
        "ta_deep_think_llm": os.getenv("TA_DEEP_LLM", "gemini-2.5-pro"),
        "ta_quick_think_llm": os.getenv("TA_QUICK_LLM", "gemini-2.5-flash"),
        "ta_max_debate_rounds": int(os.getenv("TA_DEBATE_ROUNDS", "1")),
        "ta_max_risk_rounds": int(os.getenv("TA_RISK_ROUNDS", "1")),

        # ── Bedrock (for FinMEM components) ──────────────────
        "bedrock_model": os.getenv("BEDROCK_MODEL_ID", "moonshotai.kimi-k2.5"),
        "aws_region": os.getenv("AWS_REGION", "us-east-1"),

        # ── Persistence ──────────────────────────────────────
        "state_dir": os.getenv("STATE_DIR", "./state"),
        "checkpoint_dir": os.getenv("CHECKPOINT_DIR", "./checkpoints"),
        "logs_dir": os.getenv("LOGS_DIR", "./logs"),

        # ── Memory ───────────────────────────────────────────
        "memory_top_k": int(os.getenv("MEMORY_TOP_K", "5")),
        "hmm_model_path": os.getenv("HMM_MODEL_PATH", "./models/hmm_regime.pkl"),
        "importance_model_path": os.getenv("IMPORTANCE_MODEL_PATH", "./models/importance_clf.pkl"),
    }


def get_trading_agents_config(orch_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build TradingAgents-compatible config from orchestrator config."""
    from tradingagents.default_config import DEFAULT_CONFIG

    ta_config = DEFAULT_CONFIG.copy()
    ta_config["llm_provider"] = orch_config["ta_llm_provider"]
    ta_config["deep_think_llm"] = orch_config["ta_deep_think_llm"]
    ta_config["quick_think_llm"] = orch_config["ta_quick_think_llm"]
    ta_config["max_debate_rounds"] = orch_config["ta_max_debate_rounds"]
    ta_config["max_risk_discuss_rounds"] = orch_config["ta_max_risk_rounds"]

    ta_config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    return ta_config
