"""
Autonomous Trading Orchestrator (Fully Wired)

The main loop that connects ALL components end-to-end:
    Stage 0: Sync live Alpaca positions into internal state
    Stage 1: Regime Detection (Obj1 HMM trained on SPY)
    Stage 2: Memory Decay + Importance Scoring (Obj2 Learned Importance)
    Stage 3: Decision Making (TradingAgents via Gemini, with BrainDB context)
    Stage 4: Portfolio Optimization (CVaR) + Correlation Guard (Obj3)
    Stage 5: Execution (Alpaca limit orders, stop-losses, real share sizing)
    Stage 6: Reflection & Importance Retraining
"""

import json
import os
import subprocess
import sys
import time
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

CONDA_PYTHON = os.path.expanduser("~/miniconda3/envs/tradingagents/bin/python")
TA_RUNNER = str(Path(__file__).resolve().parent.parent.parent / "TradingAgents" / "ta_runner.py")


class AutonomousTrader:
    """End-to-end autonomous trading system with all objectives wired in."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.brain = None
        self.ta_graph = None
        self.persistence = None
        self.positions_before = {}
        self.cycle_count = 0
        self._ta_subprocess_mode = False
        self.regime_classifier = None
        self.portfolio_state = {}

    def initialize(self) -> None:
        """Boot all subsystems. Call once before running cycles."""
        logger.info("=" * 60)
        logger.info("  AUTONOMOUS TRADER — INITIALIZING")
        logger.info("=" * 60)

        self._init_persistence()
        self._init_brain()
        self._init_importance_model()
        self._init_regime_classifier()
        self._init_trading_agents()
        self._load_saved_state()
        self._sync_alpaca_positions()

        logger.info("[Init] All subsystems ready")
        logger.info(f"[Init] Tickers: {self.config['portfolio_tickers']}")
        logger.info(f"[Init] TradingAgents LLM: {self.config['ta_llm_provider']} / "
                     f"{self.config['ta_deep_think_llm']}")
        logger.info(f"[Init] Cycle interval: {self.config['cycle_interval_seconds']}s")
        logger.info(f"[Init] Obj2 Importance: {'ACTIVE' if self._importance_active else 'OFF'}")

    def _init_persistence(self):
        from orchestrator.persistence import StatePersistence
        self.persistence = StatePersistence(self.config["state_dir"])

    def _init_brain(self):
        brain = self.persistence.load_brain()
        if brain is not None:
            self.brain = brain
            logger.info("[Init] BrainDB restored from checkpoint")
        else:
            from finmem.memory.layered_memory import BrainDB
            self.brain = BrainDB.create_default()
            logger.info("[Init] BrainDB created fresh")

    def _init_importance_model(self):
        """Load the Obj2 learned importance model if enabled."""
        self._importance_active = False
        if os.getenv("LEARNED_IMPORTANCE", "false").lower() == "true":
            try:
                from agentic.obj2_importance.inference import load_model, is_model_loaded
                load_model()
                self._importance_active = is_model_loaded()
                if self._importance_active:
                    logger.info("[Init] Obj2 Learned Importance model loaded")
                else:
                    logger.warning("[Init] Obj2 model file not found — using random fallback")
            except Exception as e:
                logger.warning(f"[Init] Obj2 importance load failed: {e}")

    def _init_regime_classifier(self):
        mode = self.config.get("adaptive_q_mode", "hmm")
        try:
            from agentic.obj1_regime.classifier import get_classifier
            self.regime_classifier = get_classifier(mode)
            logger.info(f"[Init] Regime classifier loaded (mode={mode})")
        except Exception as e:
            logger.warning(f"[Init] Regime classifier load failed: {e} — using threshold")
            from agentic.obj1_regime.classifier import ThresholdRegimeClassifier
            self.regime_classifier = ThresholdRegimeClassifier()

    def _init_trading_agents(self):
        self._ta_subprocess_mode = False
        try:
            from orchestrator.config import get_trading_agents_config
            from tradingagents.graph.trading_graph import TradingAgentsGraph

            ta_config = get_trading_agents_config(self.config)
            self.ta_graph = TradingAgentsGraph(
                selected_analysts=["market", "social", "news", "fundamentals"],
                debug=False, config=ta_config,
            )
            self.persistence.load_ta_memories(self.ta_graph)
            logger.info("[Init] TradingAgents graph compiled (direct)")
        except Exception as e:
            logger.warning(f"[Init] Direct TradingAgents import failed: {e}")
            if os.path.exists(CONDA_PYTHON) and os.path.exists(TA_RUNNER):
                self._ta_subprocess_mode = True
                self.ta_graph = None
                logger.info("[Init] TradingAgents will run via subprocess")
            else:
                logger.error("[Init] No TradingAgents available")
                self.ta_graph = None

    def _load_saved_state(self):
        saved = self.persistence.load_portfolio_state()
        if saved:
            self.portfolio_state = saved
            logger.info(f"[Init] Portfolio restored: ${saved.get('total_value', 0):,.2f}")
        else:
            self.portfolio_state = {
                "cash": self.config["initial_cash"],
                "positions": {},
                "total_value": self.config["initial_cash"],
                "trade_history": [],
            }

    # ──────────────────────────────────────────────────────────────────────
    #  STAGE 0: Sync Alpaca → Internal State
    # ──────────────────────────────────────────────────────────────────────

    def _sync_alpaca_positions(self):
        """Pull live Alpaca positions + account into portfolio_state.

        This ensures SELL orders use real share counts and portfolio_value
        reflects reality, not stale saved state.
        """
        try:
            from orchestrator.alpaca_enhanced import get_positions, get_account_info
            account = get_account_info()
            positions = get_positions()

            self.portfolio_state["cash"] = account["cash"]
            self.portfolio_state["total_value"] = account["portfolio_value"]
            self.portfolio_state["positions"] = positions

            pos_summary = {t: int(p["shares"]) for t, p in positions.items()}
            logger.info(f"[Sync] Alpaca → cash=${account['cash']:,.2f} | "
                        f"equity=${account['equity']:,.2f} | positions={pos_summary}")
        except Exception as e:
            logger.warning(f"[Sync] Could not sync Alpaca positions: {e}")

    # ──────────────────────────────────────────────────────────────────────
    #  STAGE 1: Regime Detection (Obj1)
    # ──────────────────────────────────────────────────────────────────────

    def detect_regime(self, ticker: str, trade_date: str) -> Dict[str, Any]:
        from agentic.obj1_regime.features import compute_features
        from agentic.obj1_regime.q_table import get_all_Q

        features = compute_features(ticker, trade_date)
        regime = self.regime_classifier.predict(features)
        try:
            proba = self.regime_classifier.predict_proba(features)
        except Exception:
            proba = {"BULL": 0.33, "SIDEWAYS": 0.34, "CRISIS": 0.33}

        q_values = get_all_Q(regime)
        logger.info(f"[Stage1] {ticker} regime={regime} | Q={q_values}")
        return {"regime": regime, "regime_proba": proba, "q_values": q_values}

    # ──────────────────────────────────────────────────────────────────────
    #  STAGE 2: Memory Decay + Importance (Obj2)
    # ──────────────────────────────────────────────────────────────────────

    def step_memory(self, trade_date: str) -> None:
        if self.brain is None:
            return
        for layer in [self.brain.short_term_memory, self.brain.mid_term_memory,
                      self.brain.long_term_memory, self.brain.reflection_memory]:
            layer.step(current_date=trade_date)
        logger.info(f"[Stage2] Memory decay step completed for {trade_date}")

    def _get_brain_context(self, ticker: str, trade_date: str, regime: str) -> str:
        """Query BrainDB for relevant memories to inject into TradingAgents prompt.

        Uses Obj2 importance scoring when available (memories with higher
        learned importance scores rank higher in retrieval).
        """
        if self.brain is None:
            return ""

        context_parts = []
        top_k = self.config.get("memory_top_k", 5)

        for layer_name, query_fn in [
            ("short_term", self.brain.query_short),
            ("mid_term", self.brain.query_mid),
            ("long_term", self.brain.query_long),
            ("reflection", self.brain.query_reflection),
        ]:
            try:
                query = f"{ticker} {trade_date} {regime} market analysis trading"
                texts, ids = query_fn(query, top_k=top_k, symbol=ticker)
                if texts:
                    context_parts.append(f"=== {layer_name.upper()} MEMORIES ===")
                    for t in texts[:top_k]:
                        context_parts.append(f"  - {t[:300]}")
            except Exception:
                pass

        if not context_parts:
            return ""

        return "\n".join(context_parts)

    # ──────────────────────────────────────────────────────────────────────
    #  STAGE 3: TradingAgents Decision
    # ──────────────────────────────────────────────────────────────────────

    def run_trading_agents(
        self, ticker: str, trade_date: str, regime: str = "SIDEWAYS",
    ) -> Dict[str, Any]:
        """Run TradingAgents with BrainDB context injected."""

        brain_context = self._get_brain_context(ticker, trade_date, regime)

        if self._ta_subprocess_mode or self.ta_graph is None:
            return self._run_ta_subprocess(ticker, trade_date, brain_context, regime)

        from orchestrator.bridge import (
            create_enriched_initial_state, ingest_ta_reports_into_brain,
        )
        enriched_state = create_enriched_initial_state(
            ticker=ticker, trade_date=trade_date,
            brain=self.brain, top_k=self.config["memory_top_k"], regime=regime,
        )
        graph_args = self.ta_graph.propagator.get_graph_args()

        logger.info(f"[Stage3] Running TradingAgents for {ticker}...")
        final_state = self.ta_graph.graph.invoke(enriched_state, **graph_args)
        self.ta_graph.curr_state = final_state

        signal = self.ta_graph.process_signal(
            final_state.get("final_trade_decision", "HOLD")
        )
        new_memory_ids = ingest_ta_reports_into_brain(
            brain=self.brain, ticker=ticker,
            trade_date=trade_date, final_state=final_state,
        )

        logger.info(f"[Stage3] {ticker} signal={signal} | new_memories={len(new_memory_ids)}")
        return {"signal": signal.strip().upper() if signal else "HOLD",
                "final_state": final_state, "memory_ids": new_memory_ids}

    def _run_ta_subprocess(
        self, ticker: str, trade_date: str,
        brain_context: str = "", regime: str = "SIDEWAYS",
    ) -> Dict[str, Any]:
        """Call TradingAgents via subprocess, passing BrainDB context as --context."""
        env = os.environ.copy()
        deep_llm = self.config.get("ta_deep_think_llm", "gemini-2.5-pro")
        quick_llm = self.config.get("ta_quick_think_llm", "gemini-2.5-flash")
        provider = self.config.get("ta_llm_provider", "google")

        # Build current portfolio summary for context
        positions = self.portfolio_state.get("positions", {})
        pos_summary = ", ".join(
            f"{t}:{int(p.get('shares', 0))}@${p.get('current_price', 0):.0f}"
            for t, p in positions.items() if p.get("shares", 0) > 0
        )
        portfolio_ctx = (
            f"Current portfolio: cash=${self.portfolio_state.get('cash', 0):,.0f}, "
            f"total=${self.portfolio_state.get('total_value', 0):,.0f}. "
            f"Positions: {pos_summary or 'none'}. "
            f"Market regime: {regime}."
        )

        full_context = f"{portfolio_ctx}\n{brain_context}" if brain_context else portfolio_ctx

        cmd = [
            CONDA_PYTHON, TA_RUNNER,
            "--ticker", ticker,
            "--date", trade_date,
            "--analysts", "market,news,fundamentals",
            "--llm-provider", provider,
            "--deep-llm", deep_llm,
            "--quick-llm", quick_llm,
            "--context", full_context[:2000],
        ]

        ta_dir = str(Path(TA_RUNNER).parent)
        logger.info(f"[Stage3] Running TradingAgents subprocess for {ticker}...")

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                env=env, cwd=ta_dir,
            )
            if proc.returncode != 0:
                logger.warning(f"[Stage3] subprocess stderr: {proc.stderr[:200]}")

            stdout = proc.stdout.strip()
            data = {}
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    break

            action = data.get("action", "HOLD").strip().upper()
            logger.info(f"[Stage3] {ticker} subprocess → {action}")

            # Ingest TA reports back into BrainDB
            if self.brain and data.get("raw_decision"):
                try:
                    mem_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
                    reports = []
                    for key in ("market_report", "news_report", "fundamentals_report"):
                        rpt = data.get(key, "")
                        if rpt:
                            reports.append(f"[{key}] {rpt[:200]}")
                    decision_text = data.get("raw_decision", "")[:300]
                    full_text = f"[{ticker}@{trade_date}] Decision={action}. {decision_text}"
                    if reports:
                        full_text += "\n" + "\n".join(reports)
                    self.brain.add_memory_short(ticker, mem_date, full_text)
                except Exception:
                    pass

            return {"signal": action, "final_state": data, "memory_ids": []}

        except subprocess.TimeoutExpired:
            logger.warning(f"[Stage3] Subprocess timeout for {ticker}")
            return {"signal": "HOLD", "final_state": {}, "memory_ids": []}
        except Exception as e:
            logger.warning(f"[Stage3] Subprocess error for {ticker}: {e}")
            return {"signal": "HOLD", "final_state": {}, "memory_ids": []}

    # ──────────────────────────────────────────────────────────────────────
    #  STAGE 4: CVaR + Correlation Guard (Obj3)
    # ──────────────────────────────────────────────────────────────────────

    def optimize_portfolio(
        self, decisions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """CVaR optimization + Obj3 concentration guard using correlation matrix."""
        from orchestrator.cvar_optimizer import apply_cvar_portfolio_optimization

        total_value = self.portfolio_state.get("total_value", 100000)

        decisions, weights = apply_cvar_portfolio_optimization(
            decisions=decisions,
            portfolio_value=total_value,
            confidence=self.config["cvar_confidence"],
            max_weight=self.config["cvar_max_weight"],
            window_days=self.config["correlation_window_days"],
            concentration_threshold=self.config["concentration_threshold"],
        )

        logger.info(f"[Stage4] CVaR weights: "
                     + ", ".join(f"{t}={w:.1%}" for t, w in weights.items()))
        return decisions

    # ──────────────────────────────────────────────────────────────────────
    #  STAGE 5: Execution (Alpaca)
    # ──────────────────────────────────────────────────────────────────────

    def execute_decisions(
        self, decisions: Dict[str, Dict[str, Any]], dry_run: bool = False,
    ) -> Dict[str, List[Dict]]:
        from orchestrator.alpaca_enhanced import (
            execute_decision, get_positions, get_account_info, is_market_open,
        )

        if dry_run:
            logger.info("[Stage5] DRY RUN — no orders submitted")
            return {t: [{"action": d.get("action", "HOLD"), "status": "dry_run"}]
                    for t, d in decisions.items()}

        if not is_market_open():
            logger.warning("[Stage5] Market is closed — skipping execution")
            return {t: [{"action": d.get("action", "HOLD"), "status": "market_closed"}]
                    for t, d in decisions.items()}

        self.positions_before = get_positions()
        execution_results = {}

        for ticker, decision in decisions.items():
            action = decision.get("action", "HOLD")
            target_shares = decision.get("target_shares", 0)

            results = execute_decision(
                ticker=ticker, action=action,
                target_shares=target_shares,
                stop_loss_pct=self.config["stop_loss_pct"],
            )
            execution_results[ticker] = results

            for r in results:
                self.persistence.log_trade({
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat(),
                    **r,
                })

        # Sync Alpaca back into internal state after execution
        try:
            account = get_account_info()
            self.portfolio_state["cash"] = account["cash"]
            self.portfolio_state["total_value"] = account["portfolio_value"]
            self.portfolio_state["positions"] = get_positions()
        except Exception as e:
            logger.warning(f"[Stage5] Account update failed: {e}")

        return execution_results

    # ──────────────────────────────────────────────────────────────────────
    #  STAGE 6: Reflection + Importance Retraining
    # ──────────────────────────────────────────────────────────────────────

    def reflect(
        self, decisions: Dict[str, Dict[str, Any]], trade_date: str,
    ) -> None:
        try:
            from orchestrator.alpaca_enhanced import get_positions
            positions_after = get_positions()
        except Exception:
            positions_after = {}

        for ticker, decision in decisions.items():
            action = decision.get("action", "HOLD")
            final_state = decision.get("final_state", {})
            memory_ids = decision.get("memory_ids", [])

            if self.ta_graph is not None:
                try:
                    from orchestrator.reflection_bridge import run_full_reflection, compute_trade_returns
                    returns = compute_trade_returns(
                        self.positions_before, positions_after, ticker
                    )
                    run_full_reflection(
                        ta_graph=self.ta_graph, brain=self.brain,
                        ticker=ticker, trade_date=trade_date,
                        decision=action, returns=returns,
                        memory_ids_used=memory_ids, final_state=final_state,
                    )
                except Exception as e:
                    logger.warning(f"[Stage6] Full reflection failed for {ticker}: {e}")
            else:
                if self.brain:
                    try:
                        mem_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
                        raw = str(final_state.get("raw_decision", ""))[:200]

                        # Compute realized P&L for this ticker if we have before/after
                        pnl_str = ""
                        before = self.positions_before.get(ticker, {})
                        after = positions_after.get(ticker, {})
                        if before.get("shares", 0) > 0 or after.get("shares", 0) > 0:
                            before_val = before.get("market_value", 0)
                            after_val = after.get("market_value", 0)
                            pnl = after_val - before_val
                            pnl_str = f" PnL=${pnl:+.2f}"

                        self.brain.add_memory_reflection(
                            ticker, mem_date,
                            f"[{ticker}@{trade_date}] Action={action}.{pnl_str} {raw}",
                        )
                        logger.info(f"[Stage6] Reflection stored for {ticker}{pnl_str}")
                    except Exception as e:
                        logger.warning(f"[Stage6] Could not store reflection for {ticker}: {e}")

            # Log reflection data for Obj2 importance retraining
            self._log_reflection_for_obj2(ticker, trade_date, action, positions_after)

        # Retrain importance model periodically
        self._maybe_retrain_importance(trade_date)

    def _log_reflection_for_obj2(
        self, ticker: str, trade_date: str,
        action: str, positions_after: Dict,
    ):
        """Log data for Obj2 importance model retraining.

        Matches the log_reflection API: date, ticker, decision, memory_ids_used,
        rationale, cumulative_return.
        """
        try:
            from agentic.obj2_importance.logger import log_reflection

            after = positions_after.get(ticker, {})
            before = self.positions_before.get(ticker, {})
            pnl = float(after.get("unrealized_pl", 0)) - float(before.get("unrealized_pl", 0))

            portfolio_val = self.portfolio_state.get("total_value", 100000)
            initial_cash = self.config.get("initial_cash", 100000)
            cum_return = (portfolio_val - initial_cash) / initial_cash if initial_cash else 0

            log_reflection(
                date=trade_date,
                ticker=ticker,
                decision=action,
                memory_ids_used=[],
                rationale=f"Action={action}, ticker PnL=${pnl:+.2f}",
                cumulative_return=cum_return,
            )
        except Exception as e:
            logger.debug(f"[Obj2] Log reflection failed for {ticker}: {e}")

    def _maybe_retrain_importance(self, trade_date: str):
        """Retrain Obj2 importance model every 5 cycles if enough data exists."""
        if not self._importance_active:
            return
        if self.cycle_count % 5 != 0:
            return

        try:
            from agentic.obj2_importance.trainer import run_training_pipeline
            from agentic.obj2_importance.inference import load_model

            for ticker in self.config["portfolio_tickers"][:1]:
                run_training_pipeline(ticker)
            load_model()
            logger.info("[Stage6] Obj2 importance model retrained")
        except Exception as e:
            logger.warning(f"[Stage6] Importance retraining failed: {e}")

    # ──────────────────────────────────────────────────────────────────────
    #  MAIN CYCLE
    # ──────────────────────────────────────────────────────────────────────

    def run_cycle(self, trade_date: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d")

        self.cycle_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"  CYCLE #{self.cycle_count} — {trade_date}")
        logger.info(f"{'='*60}")

        tickers = self.config["portfolio_tickers"]

        # Stage 0: Sync live positions from Alpaca
        self._sync_alpaca_positions()

        # Stage 2: Memory decay
        self.step_memory(trade_date)

        decisions = {}
        for ticker in tickers:
            try:
                # Stage 1: Regime
                regime_info = self.detect_regime(ticker, trade_date)
                regime = regime_info["regime"]

                # Stage 3: TradingAgents decision (with BrainDB context)
                ta_result = self.run_trading_agents(ticker, trade_date, regime)

                from orchestrator.alpaca_enhanced import get_latest_price
                price = get_latest_price(ticker)

                # Use LIVE Alpaca shares for sell sizing
                live_positions = self.portfolio_state.get("positions", {})
                current_shares = int(float(
                    live_positions.get(ticker, {}).get("shares", 0)
                ))

                decisions[ticker] = {
                    "action": ta_result["signal"],
                    "confidence": 0.7,
                    "regime": regime,
                    "price": price or 0,
                    "current_shares": current_shares,
                    "memory_ids": ta_result["memory_ids"],
                    "final_state": ta_result["final_state"],
                }

            except Exception as e:
                logger.error(f"[Cycle] Failed for {ticker}: {e}")
                decisions[ticker] = {
                    "action": "HOLD", "confidence": 0,
                    "regime": "UNKNOWN", "error": str(e),
                }

        # Stage 4: CVaR + correlation guard
        decisions = self.optimize_portfolio(decisions)

        # Stage 5: Execute
        execution_results = self.execute_decisions(decisions, dry_run=dry_run)

        # Stage 6: Reflect + log for Obj2
        self.reflect(decisions, trade_date)

        # Persist everything
        self.persistence.save_brain(self.brain)
        self.persistence.save_portfolio_state(self.portfolio_state)
        if self.ta_graph is not None:
            self.persistence.save_ta_memories(self.ta_graph)
        self.persistence.save_cycle_state({
            "cycle": self.cycle_count,
            "date": trade_date,
            "decisions": {t: d.get("action", "HOLD") for t, d in decisions.items()},
            "portfolio_value": self.portfolio_state.get("total_value", 0),
        })

        self._print_cycle_summary(trade_date, decisions, execution_results)

        return {
            "cycle": self.cycle_count,
            "date": trade_date,
            "decisions": decisions,
            "execution": execution_results,
        }

    def _print_cycle_summary(self, trade_date, decisions, execution_results):
        print(f"\n{'─'*60}")
        print(f"  Cycle #{self.cycle_count} Summary — {trade_date}")
        print(f"{'─'*60}")
        for ticker, dec in decisions.items():
            action = dec.get("action", "HOLD")
            regime = dec.get("regime", "?")
            shares = dec.get("target_shares", 0)
            weight = dec.get("cvar_weight", 0)
            override = dec.get("override_reason", "")
            exec_status = "OK"
            exec_results = execution_results.get(ticker, [])
            if exec_results:
                last = exec_results[-1] if isinstance(exec_results, list) else exec_results
                if isinstance(last, dict):
                    exec_status = last.get("status", "?")
            override_tag = f" [{override}]" if override else ""
            print(f"  {ticker:6s} | {action:10s} | regime={regime:8s} | "
                  f"shares={shares:4d} | w={weight:.2f} | exec={exec_status}{override_tag}")
        print(f"  Portfolio: ${self.portfolio_state.get('total_value', 0):,.2f}")
        print(f"{'─'*60}\n")

    # ──────────────────────────────────────────────────────────────────────
    #  AUTONOMOUS LOOP
    # ──────────────────────────────────────────────────────────────────────

    def run_autonomous(self, dry_run: bool = False) -> None:
        from orchestrator.alpaca_enhanced import is_market_open, get_next_open

        logger.info("[Autonomous] Starting autonomous trading loop")
        logger.info(f"[Autonomous] Cycle interval: {self.config['cycle_interval_seconds']}s")
        logger.info(f"[Autonomous] Dry run: {dry_run}")

        try:
            while True:
                if not dry_run and not is_market_open():
                    next_open = get_next_open()
                    if next_open:
                        logger.info(f"[Autonomous] Market closed. Next open: {next_open}")
                    else:
                        logger.info("[Autonomous] Market closed. Checking again in 60s...")
                    time.sleep(60)
                    continue

                self.run_cycle(dry_run=dry_run)

                interval = self.config["cycle_interval_seconds"]
                logger.info(f"[Autonomous] Next cycle in {interval}s (Ctrl+C to stop)")
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("\n[Autonomous] Stopped by user — saving state...")
            self.persistence.save_brain(self.brain)
            self.persistence.save_portfolio_state(self.portfolio_state)
            if self.ta_graph is not None:
                self.persistence.save_ta_memories(self.ta_graph)
            logger.info("[Autonomous] State saved. Goodbye.")
