"""
FinMEM Frontend — Flask Server

REST API backend for the FinMEM dashboard UI.
Wraps all Objective 2 and 3 modules as HTTP endpoints.

Run:
    cd FinMeM && python frontend/server.py
    Open: http://localhost:5050
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime

# Ensure project root is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("finmem-ui")

# ── CRITICAL: Set absolute paths for all data/model directories ──────────
# Instead of changing the CWD (which breaks the Flask reloader), we set
# environment variables with absolute paths. This ensures all modules
# (Obj1, Obj2, etc.) can find their files regardless of where the server starts.
os.environ["REFLECTION_LOG_DIR"] = os.path.join(project_root, "logs", "reflections")
os.environ["HMM_MODEL_PATH"] = os.path.join(project_root, "models", "hmm_regime.pkl")
os.environ["IMPORTANCE_MODEL_PATH"] = os.path.join(project_root, "models", "importance_clf.pkl")
logger.info(f"[Startup] Environment paths set relative to: {project_root}")

# Load importance model on startup
try:
    from agentic.obj2_importance.inference import load_model
    model_path = os.path.join(project_root, "models", "importance_clf.pkl")
    if os.path.exists(model_path):
        load_model(model_path)
        logger.info(f"[Startup] Importance model loaded from {model_path}")
    else:
        logger.warning(f"[Startup] Importance model not found at {model_path} — run 'Train on Real Reflections' in dashboard")
except Exception as e:
    logger.warning(f"[Startup] Could not load importance model: {e}")


# ── Static Files ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


@app.route("/old")
def old_index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)


# ── Status ────────────────────────────────────────────────────────────────

@app.route("/api/status")
def system_status():
    """System status and configuration."""
    status = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "llm_provider": os.getenv("LLM_PROVIDER", "bedrock"),
            "bedrock_model": os.getenv("BEDROCK_MODEL_ID", "N/A"),
            "adaptive_q": os.getenv("ADAPTIVE_Q", "false"),
            "learned_importance": os.getenv("LEARNED_IMPORTANCE", "false"),
            "cross_ticker": os.getenv("CROSS_TICKER", "false"),
            "portfolio_tickers": os.getenv("PORTFOLIO_TICKERS", "TSLA,NVDA,MSFT,AMZN,NFLX"),
        },
        "objectives": {
            "obj1": {"name": "Adaptive Q", "status": "active" if os.getenv("ADAPTIVE_Q", "false").lower() == "true" else "inactive"},
            "obj2": {"name": "Learned Importance", "status": "active" if os.getenv("LEARNED_IMPORTANCE", "false").lower() == "true" else "inactive"},
            "obj3": {"name": "Cross-Ticker", "status": "active" if os.getenv("CROSS_TICKER", "false").lower() == "true" else "inactive"},
        }
    }

    # Check importance model
    try:
        from agentic.obj2_importance.inference import get_model_info
        status["importance_model"] = get_model_info()
    except Exception:
        status["importance_model"] = {"loaded": False}

    return jsonify(status)


# ── Objective 2 Endpoints ─────────────────────────────────────────────────

@app.route("/api/test-obj2", methods=["POST"])
def test_obj2():
    """Train Obj2 importance classifier on REAL reflection logs for selected ticker."""
    try:
        data = request.get_json(silent=True) or {}
        ticker = data.get("ticker", "TSLA").upper()
        
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            from agentic.obj2_importance.trainer import run_training_pipeline
            run_training_pipeline(ticker)

        output = f.getvalue()

        # Reload model after training
        from agentic.obj2_importance.inference import load_model, get_model_info
        load_model(os.path.join(project_root, "models", "importance_clf.pkl"))
        info = get_model_info()

        return jsonify({"success": True, "output": output, "model_info": info})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


@app.route("/api/train-importance", methods=["POST"])
def train_importance():
    """Train the importance classifier."""
    try:
        data = request.json or {}
        ticker = data.get("ticker", "TSLA")

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            from agentic.obj2_importance.trainer import run_training_pipeline
            run_training_pipeline(ticker)

        output = f.getvalue()

        # Get model info after training
        from agentic.obj2_importance.inference import get_model_info, load_model
        load_model()
        info = get_model_info()

        return jsonify({"success": True, "output": output, "model_info": info})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


@app.route("/api/importance-score", methods=["POST"])
def importance_score():
    """Get importance score for given features."""
    try:
        data = request.json or {}
        from agentic.obj2_importance.inference import get_importance_score

        score = get_importance_score(
            layer=data.get("layer", "short"),
            age_days=int(data.get("age_days", 0)),
            access_count=int(data.get("access_count", 0)),
            text_length=int(data.get("text_length", 100)),
            sentiment_score=float(data.get("sentiment_score", 0.0)),
        )

        return jsonify({"success": True, "score": score})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/reflections")
def get_reflections():
    """Load all reflection logs."""
    try:
        from agentic.obj2_importance.logger import load_all_reflections
        reflections = load_all_reflections()
        return jsonify({"success": True, "count": len(reflections), "reflections": reflections[-50:]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ── Objective 3 Endpoints ─────────────────────────────────────────────────

@app.route("/api/test-obj3", methods=["POST"])
def test_obj3():
    """Run Objective 3 end-to-end test."""
    try:
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            from agentic.obj3_correlation.test_correlation import test_objective_3
            test_objective_3()

        output = f.getvalue()
        return jsonify({"success": True, "output": output})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


@app.route("/api/correlation-matrix", methods=["POST", "GET"])
def correlation_matrix():
    """Compute and return the correlation matrix."""
    try:
        data = request.get_json(silent=True) or {}
        tickers_str = data.get("tickers", os.getenv("PORTFOLIO_TICKERS", "TSLA,NVDA,MSFT,AMZN,NFLX"))

        if isinstance(tickers_str, str):
            tickers = [t.strip() for t in tickers_str.split(",")]
        else:
            tickers = tickers_str

        from agentic.obj3_correlation.matrix import compute_correlation_matrix
        corr = compute_correlation_matrix(tickers)

        # Convert to serializable format
        matrix_data = {}
        for t1 in tickers:
            matrix_data[t1] = {}
            for t2 in tickers:
                try:
                    matrix_data[t1][t2] = round(float(corr.loc[t1, t2]), 4)
                except (KeyError, ValueError):
                    matrix_data[t1][t2] = 0.0

        return jsonify({
            "success": True,
            "tickers": tickers,
            "matrix": matrix_data,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


@app.route("/api/test-guard", methods=["POST", "GET"])
def test_guard():
    """Test the concentration guard with mock decisions."""
    try:
        data = request.get_json(silent=True) or {}
        import pandas as pd
        from agentic.obj3_correlation.guard import apply_concentration_guard, get_guard_summary

        decisions = data.get("decisions", {
            "TSLA": {"action": "BUY", "confidence": 0.78},
            "NVDA": {"action": "BUY", "confidence": 0.71},
            "MSFT": {"action": "BUY", "confidence": 0.65},
            "AMZN": {"action": "HOLD", "confidence": 0.50},
            "NFLX": {"action": "SELL", "confidence": 0.60},
        })

        threshold = float(data.get("threshold", 0.80))

        # Get or mock correlation matrix
        try:
            from agentic.obj3_correlation.matrix import compute_correlation_matrix
            tickers = list(decisions.keys())
            corr = compute_correlation_matrix(tickers)
        except Exception:
            tickers = list(decisions.keys())
            import numpy as np
            n = len(tickers)
            corr_data = np.eye(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        corr_data[i][j] = 0.5 + 0.3 * (1 if abs(i - j) == 1 else 0)
            corr = pd.DataFrame(corr_data, index=tickers, columns=tickers)

        modified, trigger_count = apply_concentration_guard(decisions, corr, threshold)
        summary = get_guard_summary(modified)

        return jsonify({
            "success": True,
            "decisions": modified,
            "trigger_count": trigger_count,
            "summary": summary,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


# ── Simulation Endpoint ───────────────────────────────────────────────────

@app.route("/api/run-simulation", methods=["POST"])
def run_simulation():
    """Run a trading simulation."""
    try:
        data = request.json or {}
        ticker = data.get("ticker", "TSLA")
        mode = data.get("mode", "train")
        start_date = data.get("start_date", "2022-03-14")
        end_date = data.get("end_date", "2022-06-15")
        capital = float(data.get("capital", 100000))

        # Set objective flags
        if data.get("adaptive_q", True):
            os.environ["ADAPTIVE_Q"] = "true"
        if data.get("learned_importance", True):
            os.environ["LEARNED_IMPORTANCE"] = "true"
        if data.get("cross_ticker", False):
            os.environ["CROSS_TICKER"] = "true"

        from finmem.simulation.simulator import TradingSimulator
        from finmem.config import FinMEMConfig

        config = FinMEMConfig()
        config.initial_capital = capital

        simulator = TradingSimulator(config)

        from datetime import datetime as dt
        result = simulator.run(
            ticker=ticker,
            start_date=dt.strptime(start_date, "%Y-%m-%d").date(),
            end_date=dt.strptime(end_date, "%Y-%m-%d").date(),
            mode=mode,
            initial_capital=capital,
            verbose=False,
        )

        return jsonify({
            "success": True,
            "ticker": result.ticker,
            "mode": result.mode,
            "days_processed": result.days_processed,
            "initial_capital": result.initial_capital,
            "final_value": result.final_value,
            "total_return": result.total_return,
            "total_return_pct": result.total_return_pct,
            "metrics": result.metrics,
            "bh_metrics": result.bh_metrics,
            "memory_stats": result.memory_stats,
            "trade_count": len([t for t in result.trades if t.get("action") != "HOLD"]),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


@app.route("/api/ablation-results", methods=["GET"])
def get_ablation_results():
    """Get list of ablation results and their content."""
    try:
        import glob
        import csv
        results_files = glob.glob(os.path.join(project_root, "artifacts", "ablation_results*.csv"))
        
        all_results = {}
        for path in results_files:
            filename = os.path.basename(path)
            label = filename.replace("ablation_results_", "").replace(".csv", "")
            if label == "ablation_results":
                label = "default"
            
            data = []
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            all_results[label] = data
            
        return jsonify({"success": True, "results": all_results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


# ── Objective 1 Endpoints ─────────────────────────────────────────────────

@app.route("/api/regime-classify", methods=["POST"])
def regime_classify():
    """Classify market regime for a given ticker using both classifiers."""
    try:
        data = request.json or {}
        ticker = data.get("ticker", "TSLA")

        import yfinance as yf
        import numpy as np
        from agentic.obj1_regime.classifier import get_classifier

        price_data = yf.download(ticker, period="60d", progress=False)
        if price_data.empty:
            return jsonify({"success": False, "error": f"No price data for {ticker}"})

        returns = price_data['Close'].pct_change().dropna().values.flatten()
        vol = float(np.std(returns[-20:]))
        ret = float(np.mean(returns[-20:]))
        
        # For HMM we specifically need the sequence of log returns
        close = price_data['Close'].values.flatten()
        log_returns = np.log(close[1:] / close[:-1])
        returns_seq = log_returns.tolist()
        
        features = {
            "volatility": vol, 
            "return": ret,
            "returns_seq": returns_seq
        }

        # Threshold classifier
        tc = get_classifier("threshold")
        t_regime = tc.predict(features)
        t_probs = tc.predict_proba(features)

        # HMM classifier
        hc = get_classifier("hmm")
        h_regime = hc.predict(features)
        h_probs = hc.predict_proba(features)

        # Adaptive Q values based on regime
        q_map = {
            "BULL": {"shallow": 10, "intermediate": 60, "deep": 240},
            "SIDEWAYS": {"shallow": 5, "intermediate": 45, "deep": 180},
            "CRISIS": {"shallow": 2, "intermediate": 20, "deep": 90},
        }

        return jsonify({
            "success": True,
            "ticker": ticker,
            "features": {"volatility_20d": round(vol, 6), "mean_return_20d": round(ret, 6)},
            "threshold": {"regime": t_regime, "probabilities": {k: round(v, 4) for k, v in t_probs.items()}},
            "hmm": {"regime": h_regime, "probabilities": {k: round(v, 4) for k, v in h_probs.items()}},
            "adaptive_q": q_map.get(h_regime, q_map["SIDEWAYS"]),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


# ── Objective 4 Endpoints ─────────────────────────────────────────────────

@app.route("/api/obj4-info", methods=["GET"])
def obj4_info():
    """Get Obj4 multi-agent pipeline information."""
    try:
        from agentic.obj4_multiagent.graph import build_graph
        g = build_graph()
        nodes = list(g.nodes)

        agent_models = {
            "fundamental": os.getenv("FUNDAMENTAL_MODEL", "N/A"),
            "sentiment": os.getenv("SENTIMENT_MODEL", "N/A"),
            "technical": os.getenv("TECHNICAL_MODEL", "N/A"),
            "regime": os.getenv("REGIME_MODEL", "N/A"),
            "debate": os.getenv("DEBATE_MODEL", "N/A"),
            "portfolio": os.getenv("PORTFOLIO_MODEL", "N/A"),
        }

        return jsonify({
            "success": True,
            "nodes": nodes,
            "node_count": len(nodes),
            "pipeline": [
                {"step": 1, "node": "regime_node_agent", "description": "Classifies market regime (BULL/SIDEWAYS/CRISIS)"},
                {"step": 2, "node": "fundamental", "description": "Analyzes long-term & mid-term memories for value signals"},
                {"step": 2, "node": "sentiment", "description": "Analyzes short-term memories for market mood"},
                {"step": 2, "node": "technical", "description": "Analyzes price history for momentum & patterns"},
                {"step": 3, "node": "debate", "description": "Bull vs Bear structured debate (2 rounds)"},
                {"step": 4, "node": "risk", "description": "Risk manager applies regime-adjusted confidence"},
                {"step": 5, "node": "final", "description": "Portfolio manager makes final BUY/HOLD/SELL decision"},
            ],
            "agent_models": agent_models,
            "debate_rounds": int(os.getenv("DEBATE_ROUNDS", "2")),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})



# ── Portfolio & Alpaca Endpoints ──────────────────────────────────────────

@app.route("/api/portfolio", methods=["GET"])
def get_portfolio():
    """Get live Alpaca portfolio: account info + all positions."""
    try:
        sys.path.insert(0, os.path.join(project_root, "..", "TradingAgents"))
        from orchestrator.alpaca_enhanced import get_account_info, get_positions
        account = get_account_info()
        positions = get_positions()
        return jsonify({"success": True, "account": account, "positions": positions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ── Backtest Endpoint ─────────────────────────────────────────────────────

@app.route("/api/backtest", methods=["POST"])
def run_backtest_endpoint():
    """Run a backtest over a historical date range."""
    try:
        sys.path.insert(0, os.path.join(project_root, "..", "TradingAgents"))
        data = request.json or {}
        tickers_str = data.get("tickers", os.getenv("PORTFOLIO_TICKERS", "TSLA,NVDA,MSFT,AMZN,AAPL"))
        if isinstance(tickers_str, str):
            tickers = [t.strip() for t in tickers_str.split(",")]
        else:
            tickers = tickers_str

        start_date = data.get("start_date", "2024-01-01")
        end_date = data.get("end_date", "2024-03-01")
        initial_cash = float(data.get("capital", 100000))
        use_agents = data.get("use_agents", False)

        from orchestrator.backtester import run_backtest
        result = run_backtest(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            use_trading_agents=use_agents,
        )

        return jsonify({"success": True, **result.to_dict()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


# ── Live Cycle Endpoint ───────────────────────────────────────────────────

@app.route("/api/run-cycle", methods=["POST"])
def run_live_cycle():
    """Run a single autonomous trading cycle (dry-run by default)."""
    try:
        sys.path.insert(0, os.path.join(project_root, "..", "TradingAgents"))
        data = request.json or {}
        dry_run = data.get("dry_run", True)

        from orchestrator.config import get_orchestrator_config
        from orchestrator.orchestrator import AutonomousTrader

        config = get_orchestrator_config()
        trader = AutonomousTrader(config)
        trader.initialize()
        result = trader.run_cycle(dry_run=dry_run)

        decisions_clean = {}
        for ticker, dec in result.get("decisions", {}).items():
            decisions_clean[ticker] = {
                k: v for k, v in dec.items()
                if k not in ("final_state",) and not callable(v)
            }

        return jsonify({
            "success": True,
            "cycle": result.get("cycle"),
            "date": result.get("date"),
            "decisions": decisions_clean,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


# ── CVaR Weights Endpoint ────────────────────────────────────────────────

@app.route("/api/cvar-weights", methods=["POST"])
def cvar_weights():
    """Compute CVaR-optimal portfolio weights for given tickers."""
    try:
        sys.path.insert(0, os.path.join(project_root, "..", "TradingAgents"))
        data = request.json or {}
        tickers_str = data.get("tickers", os.getenv("PORTFOLIO_TICKERS", "TSLA,NVDA,MSFT,AMZN,AAPL"))
        if isinstance(tickers_str, str):
            tickers = [t.strip() for t in tickers_str.split(",")]
        else:
            tickers = tickers_str

        from orchestrator.cvar_optimizer import _fetch_returns, optimize_cvar_weights
        returns_df = _fetch_returns(tickers, window_days=60)
        weights = optimize_cvar_weights(tickers, returns_df, 0.95, 0.35)

        return jsonify({"success": True, "weights": weights, "tickers": tickers})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ── Pipeline Status Endpoint ──────────────────────────────────────────────

@app.route("/api/pipeline-status", methods=["GET"])
def pipeline_status():
    """Show exactly which components are wired, active, and what data flows where.

    This is the 'product health check' — it tells the user what
    the system will actually do when a cycle runs.
    """
    status = {"stages": [], "issues": []}

    # Stage 0: Alpaca Sync
    alpaca_ok = False
    try:
        from orchestrator.alpaca_enhanced import get_account_info, get_positions
        account = get_account_info()
        positions = get_positions()
        alpaca_ok = True
        pos_list = [{"ticker": t, "shares": int(float(p.get("shares", 0))),
                     "market_value": round(float(p.get("market_value", 0)), 2)}
                    for t, p in positions.items()]
        status["stages"].append({
            "stage": 0, "name": "Alpaca Portfolio Sync",
            "status": "active", "wired": True,
            "detail": f"Cash=${account['cash']:,.2f}, "
                      f"Equity=${account['equity']:,.2f}, "
                      f"{len(positions)} positions",
            "positions": pos_list,
        })
    except Exception as e:
        status["stages"].append({
            "stage": 0, "name": "Alpaca Portfolio Sync",
            "status": "error", "wired": True, "detail": str(e),
        })
        status["issues"].append("Alpaca API not reachable — SELL orders cannot be sized correctly")

    # Stage 1: Regime Detection (Obj1)
    regime_ok = False
    try:
        from agentic.obj1_regime.classifier import get_classifier
        clf = get_classifier("hmm")
        regime_ok = True
        status["stages"].append({
            "stage": 1, "name": "Regime Detection (Obj1 HMM)",
            "status": "active", "wired": True,
            "detail": "HMM classifier trained on SPY — classifies BULL/SIDEWAYS/CRISIS per ticker",
            "feeds_into": ["Stage 3 (TradingAgents context)", "Stage 4 (CVaR risk adjustment)"],
        })
    except Exception as e:
        status["stages"].append({
            "stage": 1, "name": "Regime Detection (Obj1)",
            "status": "fallback", "wired": True,
            "detail": f"HMM load failed ({e}), using threshold classifier",
        })

    # Stage 2: Memory + Learned Importance (Obj2)
    importance_active = os.getenv("LEARNED_IMPORTANCE", "false").lower() == "true"
    model_loaded = False
    try:
        from agentic.obj2_importance.inference import is_model_loaded, get_model_info
        model_loaded = is_model_loaded()
        model_info = get_model_info()
    except Exception:
        model_info = {"loaded": False}

    status["stages"].append({
        "stage": 2, "name": "Memory Decay + Learned Importance (Obj2)",
        "status": "active" if importance_active and model_loaded else "partial",
        "wired": True,
        "detail": (
            f"BrainDB 4-layer memory (short/mid/long/reflection). "
            f"Obj2 importance: {'ACTIVE (model loaded)' if model_loaded else 'OFF — using random fallback'}. "
            f"Memories are queried per-ticker and injected into TradingAgents context."
        ),
        "importance_model": model_info,
        "feeds_into": ["Stage 3 (BrainDB context → TA subprocess --context arg)"],
        "auto_retrain": "Every 5 cycles" if importance_active else "Disabled",
    })

    # Stage 3: TradingAgents Decision
    ta_mode = "unknown"
    ta_detail = ""
    try:
        conda_python = os.path.expanduser("~/miniconda3/envs/tradingagents/bin/python")
        ta_runner = os.path.join(project_root, "..", "TradingAgents", "ta_runner.py")
        if os.path.exists(conda_python) and os.path.exists(ta_runner):
            ta_mode = "subprocess"
            ta_detail = (
                f"Runs via subprocess in tradingagents conda env. "
                f"LLM: {os.getenv('TA_LLM_PROVIDER', 'google')} / "
                f"{os.getenv('TA_DEEP_LLM', 'gemini-2.5-pro')}. "
                f"BrainDB context + portfolio state injected via --context flag."
            )
        else:
            ta_mode = "missing"
            ta_detail = "Neither subprocess nor direct import available"
            status["issues"].append("TradingAgents not available")
    except Exception:
        pass

    status["stages"].append({
        "stage": 3, "name": "TradingAgents Decision",
        "status": "active" if ta_mode == "subprocess" else "error",
        "wired": True,
        "mode": ta_mode,
        "detail": ta_detail,
        "receives_from": [
            "Stage 0 (current portfolio positions + cash)",
            "Stage 1 (regime label)",
            "Stage 2 (BrainDB memories ranked by importance)",
        ],
        "feeds_into": ["Stage 4 (BUY/SELL/HOLD signals)"],
    })

    # Stage 4: CVaR + Correlation Guard (Obj3)
    status["stages"].append({
        "stage": 4, "name": "CVaR Portfolio Optimization + Correlation Guard (Obj3)",
        "status": "active", "wired": True,
        "detail": (
            f"CVaR confidence={os.getenv('CVAR_CONFIDENCE', '0.95')}, "
            f"max_weight={os.getenv('CVAR_MAX_WEIGHT', '0.35')}, "
            f"correlation_threshold={os.getenv('CONCENTRATION_THRESHOLD', '0.80')}. "
            f"Computes optimal weights via tail-risk minimization. "
            f"Correlation guard overrides lower-confidence BUY→HOLD when pairwise correlation > threshold."
        ),
        "receives_from": [
            "Stage 3 (ticker decisions with confidence)",
            "Stage 0 (portfolio_value for position sizing)",
        ],
        "feeds_into": ["Stage 5 (sized orders: target_shares per ticker)"],
    })

    # Stage 5: Execution
    status["stages"].append({
        "stage": 5, "name": "Alpaca Execution",
        "status": "active" if alpaca_ok else "error",
        "wired": True,
        "detail": (
            "Executes BUY (limit orders), SELL (market orders), stop-loss placement. "
            "SELL uses real Alpaca share count (synced from Stage 0). "
            "After execution, portfolio_state is re-synced from Alpaca."
        ),
        "receives_from": ["Stage 4 (action + target_shares per ticker)"],
        "feeds_into": ["Stage 6 (positions_before vs positions_after for P&L)"],
    })

    # Stage 6: Reflection
    status["stages"].append({
        "stage": 6, "name": "Reflection + Obj2 Retraining",
        "status": "active", "wired": True,
        "detail": (
            "Stores reflection in BrainDB reflection layer (ticker, date, action, P&L). "
            "Logs data to logs/reflections/*.jsonl for Obj2 importance retraining. "
            "Auto-retrains importance model every 5 cycles (if enabled)."
        ),
        "receives_from": [
            "Stage 5 (execution results, positions_before/after)",
            "Stage 3 (agent reports)",
        ],
        "feeds_into": [
            "Stage 2 (BrainDB reflection memories for future cycles)",
            "Obj2 importance model (retraining data)",
        ],
    })

    # Summary
    wired_count = sum(1 for s in status["stages"] if s.get("wired"))
    active_count = sum(1 for s in status["stages"] if s.get("status") == "active")
    status["summary"] = {
        "total_stages": len(status["stages"]),
        "wired": wired_count,
        "active": active_count,
        "tickers": os.getenv("PORTFOLIO_TICKERS", "TSLA,NVDA,MSFT,AMZN,AAPL").split(","),
        "fully_wired": len(status["issues"]) == 0,
    }

    return jsonify({"success": True, **status})


if __name__ == "__main__":
    print("\n  FinMEM + TradingAgents Dashboard")
    print("  ────────────────────────────────────")
    print("  Open: http://localhost:5050")
    print("  ────────────────────────────────────\n")
    app.run(host="0.0.0.0", port=5050, debug=True)
