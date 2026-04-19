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


# ── Static Files ──────────────────────────────────────────────────────────

@app.route("/")
def index():
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
    """Run Objective 2 end-to-end test."""
    try:
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            from agentic.obj2_importance.test_importance import test_objective_2
            test_objective_2()

        output = f.getvalue()
        return jsonify({"success": True, "output": output})
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


@app.route("/api/correlation-matrix", methods=["POST"])
def correlation_matrix():
    """Compute and return the correlation matrix."""
    try:
        data = request.json or {}
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


@app.route("/api/test-guard", methods=["POST"])
def test_guard():
    """Test the concentration guard with mock decisions."""
    try:
        data = request.json or {}
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


if __name__ == "__main__":
    print("\n  🚀 FinMEM Dashboard Server")
    print("  ────────────────────────────────────")
    print("  Open: http://localhost:5050")
    print("  ────────────────────────────────────\n")
    app.run(host="0.0.0.0", port=5050, debug=True)
