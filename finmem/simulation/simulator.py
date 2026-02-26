"""
Trading Simulator — Paper-Faithful Implementation

The main agent class that orchestrates the FinMEM trading loop.
Processes one trading day at a time through the step() method:

1. Receive market info from environment
2. Summarize news via LLM (working memory: summarization)
3. Store summarized data in appropriate memory layers
4. Update self-adaptive character based on 3-day return
5. Run reflection (LLM-based analysis)
6. Execute trade decision (1 share per paper)
7. Update access counters from portfolio feedback
8. Memory system step (decay, cleanup, jumps)
"""

import os
import pickle
import logging
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field

from ..config import FinMEMConfig, DEFAULT_CONFIG
from ..memory.layered_memory import BrainDB
from ..llm_client import LLMClient
from ..decision.reflection import trading_reflection, summarize_news, observe_price
from ..profiling.agent_profile import AgentProfile
from ..evaluation.metrics import compute_metrics, compute_buy_and_hold, format_metrics_report
from .environment import MarketEnvironment
from .portfolio import Portfolio
from ..data.build_dataset import build_dataset, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    ticker: str
    mode: str
    start_date: date
    end_date: date
    days_processed: int
    initial_capital: float
    final_value: float
    total_return: float
    total_return_pct: float
    trades: List[Dict[str, Any]]
    reflection_results: Dict[date, Dict[str, Any]]
    memory_stats: Dict[str, Any]
    # Paper metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    bh_metrics: Dict[str, Any] = field(default_factory=dict)


class TradingSimulator:
    """FinMEM Trading Agent — Paper-Faithful Implementation.
    
    Orchestrates the full trading pipeline per the paper:
    - Self-adaptive character profiling (switches risk on 3-day return)
    - Memory system (BrainDB with 4 layers, paper-exact decay)
    - Working memory operations (summarization → reflection)
    - LLM-based promotion of pivotal memories (+5 bonus)
    - Single-share portfolio management
    - 5-metric evaluation (Sharpe, CR, Vol, Drawdown)
    """

    def __init__(
        self,
        config: Optional[FinMEMConfig] = None,
        brain: Optional[BrainDB] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        
        # Initialize LLM client
        self.llm = LLMClient(config=self.config.llm)
        
        # Initialize memory system
        self.brain = brain or BrainDB.from_config(self.config.get_brain_config())
        
        # Initialize self-adaptive agent profile (paper's key innovation)
        self.profile = AgentProfile.create_self_adaptive()
        
        # Simulation state
        self.portfolio: Optional[Portfolio] = None
        self.reflection_results: Dict[date, Dict[str, Any]] = {}
        self.day_counter = 0

    def step(
        self,
        cur_date: date,
        cur_price: float,
        filing_k: Optional[str],
        filing_q: Optional[str],
        news: List[str],
        run_mode: str,
        future_record: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process one trading day — paper's core loop.
        
        Args:
            cur_date: Current trading date.
            cur_price: Current stock price.
            filing_k: Annual filing text (if any).
            filing_q: Quarterly filing text (if any).
            news: List of news headlines/summaries.
            run_mode: "train" or "test".
            future_record: Next-day price change (train mode only).
            
        Returns:
            Reflection result dictionary.
        """
        symbol = self.portfolio.symbol if self.portfolio else "UNKNOWN"
        
        # 1. Store filings in memory (mid and long layers)
        if filing_q:
            self.brain.add_memory_mid(
                symbol=symbol, mem_date=cur_date, text=filing_q
            )
        if filing_k:
            self.brain.add_memory_long(
                symbol=symbol, mem_date=cur_date, text=filing_k
            )
        
        # 2. Working Memory Op 1: Summarize news via LLM before storage
        if news:
            summarized = summarize_news(
                llm=self.llm,
                news_list=news,
                symbol=symbol,
                cur_date=cur_date,
            )
            if summarized:
                self.brain.add_memory_short(
                    symbol=symbol, mem_date=cur_date, text=summarized
                )
        
        # 3. Update portfolio with current price
        self.portfolio.update_market_info(
            new_market_price_info=cur_price,
            cur_date=cur_date,
        )
        
        # 4. Self-adaptive character: update risk mode based on 3-day return
        moment_data = self.portfolio.get_moment(moment_window=3)
        momentum_val = None
        cum_3day_return = 0
        if moment_data is not None:
            momentum_val = moment_data["moment"]
            # Compute actual 3-day cumulative return for character update
            if len(self.portfolio.price_history) >= 4:
                p_now = self.portfolio.price_history[-1]
                p_3ago = self.portfolio.price_history[-4]
                cum_3day_return = (p_now - p_3ago) / p_3ago if p_3ago > 0 else 0
            self.profile.update_character(cum_3day_return)
        
        # 5. Working Memory Op 2: Observation — structured price analysis via LLM
        observation = observe_price(
            llm=self.llm,
            symbol=symbol,
            cur_date=cur_date,
            cur_price=cur_price,
            price_history=self.portfolio.price_history,
            momentum=momentum_val,
        )
        if observation:
            self.brain.add_memory_short(
                symbol=symbol, mem_date=cur_date, text=observation
            )
        
        # 5. Get dynamic character string for memory query
        character_string = self.profile.get_character_string(symbol)
        
        # 7. Run reflection (core working memory operation 3)
        momentum = None
        if run_mode == "test" and momentum_val is not None:
            momentum = momentum_val
        
        reflection_result = trading_reflection(
            cur_date=cur_date,
            symbol=symbol,
            brain=self.brain,
            llm=self.llm,
            character_string=character_string,
            top_k=self.config.memory.top_k,
            run_mode=run_mode,
            future_record=future_record,
            momentum=momentum,
        )
        
        self.reflection_results[cur_date] = reflection_result
        
        # 7. Construct and execute action (1 share per paper)
        if run_mode == "train":
            direction = 1 if (future_record and future_record > 0) else -1
            action = {"direction": direction}
        else:
            decision = reflection_result.get("investment_decision", "hold")
            if decision == "buy":
                action = {"direction": 1}
            elif decision == "sell":
                action = {"direction": -1}
            else:
                action = {"direction": 0}
        
        # 8. Execute the action
        self.portfolio.record_action(action)
        self.portfolio.update_portfolio_series()
        
        # 9. Update access counters based on portfolio feedback
        self._update_access_counters()
        
        # 10. Memory system step (decay, cleanup, jumps)
        self.brain.step()
        
        self.day_counter += 1
        
        logger.info(
            f"[Day {self.day_counter}] {cur_date} | "
            f"Price: ${cur_price:.2f} | "
            f"Action: {action.get('direction', 0)} | "
            f"Character: {self.profile.current_risk_mode} | "
            f"Portfolio: {self.portfolio.get_summary()}"
        )
        
        return reflection_result

    def _update_access_counters(self) -> None:
        """Update memory access counters based on portfolio feedback."""
        feedback = self.portfolio.get_feedback_response()
        if not feedback or feedback["feedback"] == 0:
            return
        
        feedback_date = feedback["date"]
        if feedback_date not in self.reflection_results:
            return
        
        reflection = self.reflection_results[feedback_date]
        all_ids = reflection.get("_all_memory_ids", {})
        
        all_mem_ids = []
        for layer in ["short", "mid", "long", "reflection"]:
            all_mem_ids.extend(all_ids.get(layer, []))
        
        if all_mem_ids:
            self.brain.update_access_count_with_feedback(
                symbol=self.portfolio.symbol,
                ids=all_mem_ids,
                feedback=feedback["feedback"],
            )

    def run(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        mode: str = "train",
        initial_capital: float = 100000.0,
        dataset_path: Optional[str] = None,
        verbose: bool = True,
    ) -> SimulationResult:
        """Run a full simulation.
        
        Args:
            ticker: Stock ticker to trade.
            start_date: Simulation start date.
            end_date: Simulation end date.
            mode: "train" or "test".
            initial_capital: Starting cash.
            dataset_path: Path to pre-built dataset.
            verbose: Print progress.
            
        Returns:
            SimulationResult with trades, metrics, and memory stats.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  FinMEM Trading Simulation (Paper-Faithful)")
            print(f"  Ticker: {ticker} | Mode: {mode}")
            print(f"  Period: {start_date} → {end_date}")
            print(f"  Capital: ${initial_capital:,.2f}")
            print(f"  Character: Self-Adaptive (paper default)")
            print(f"{'='*60}\n")
        
        # Load or build dataset
        if dataset_path and os.path.exists(dataset_path):
            dataset = load_dataset(dataset_path)
            if verbose:
                print(f"  Loaded dataset: {len(dataset)} days")
        else:
            if verbose:
                print(f"  Building dataset from Yahoo Finance...")
            dataset = build_dataset(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            if verbose:
                print(f"  Built dataset: {len(dataset)} days")
        
        if not dataset:
            raise ValueError(f"No data available for {ticker} in the given date range")
        
        # Initialize environment
        env = MarketEnvironment(
            env_data=dataset,
            start_date=start_date,
            end_date=end_date,
            symbol=ticker,
        )
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            symbol=ticker,
            lookback_window_size=self.config.look_back_window_size,
            cash=initial_capital,
        )
        
        # Reset state
        self.reflection_results = {}
        self.day_counter = 0
        self.profile = AgentProfile.create_self_adaptive()
        
        # Collect prices for B&H baseline
        prices_for_bh = []
        
        if verbose:
            print(f"\n  Starting {mode} simulation ({env.simulation_length} days)...\n")
        
        # Main simulation loop
        while True:
            result = env.step()
            cur_date, cur_price, filing_k, filing_q, news, future_record, terminated = result
            
            if terminated:
                break
            
            prices_for_bh.append(cur_price)
            
            if verbose and self.day_counter % 5 == 0:
                print(f"  Day {self.day_counter + 1}: {cur_date} | "
                      f"${cur_price:.2f} | "
                      f"[{self.profile.current_risk_mode}] | "
                      f"{self.portfolio.get_summary()}")
            
            self.step(
                cur_date=cur_date,
                cur_price=cur_price,
                filing_k=filing_k,
                filing_q=filing_q,
                news=news,
                run_mode=mode,
                future_record=future_record if mode == "train" else None,
            )
        
        # Calculate results
        final_value = self.portfolio.total_value
        total_return = final_value - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Compute paper's 5 evaluation metrics
        agent_metrics = compute_metrics(self.portfolio.portfolio_value_history)
        bh_metrics = compute_buy_and_hold(prices_for_bh, initial_capital)
        
        result = SimulationResult(
            ticker=ticker,
            mode=mode,
            start_date=start_date,
            end_date=end_date,
            days_processed=self.day_counter,
            initial_capital=initial_capital,
            final_value=final_value,
            total_return=total_return,
            total_return_pct=total_return_pct,
            trades=self.portfolio.action_history,
            reflection_results=self.reflection_results,
            memory_stats=self.brain.stats(ticker),
            metrics=agent_metrics,
            bh_metrics=bh_metrics,
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Simulation Complete")
            print(f"  Days Processed: {self.day_counter}")
            print(f"  Final Value:    ${final_value:,.2f}")
            print(f"  Total Return:   ${total_return:,.2f} ({total_return_pct:+.2f}%)")
            print(f"\n  📊 Evaluation Metrics (Paper's 5 Metrics):")
            print(f"  {'─'*50}")
            print(format_metrics_report(agent_metrics, bh_metrics, ticker))
            print(f"\n  Memory Stats:   {self.brain.stats(ticker)}")
            print(f"{'='*60}\n")
        
        return result

    # ── Checkpointing ──

    def save_checkpoint(self, path: str) -> None:
        """Save full agent state for resume."""
        os.makedirs(path, exist_ok=True)
        
        self.brain.save_checkpoint(os.path.join(path, "brain"))
        
        state = {
            "config": self.config,
            "portfolio": self.portfolio,
            "profile": self.profile,
            "reflection_results": self.reflection_results,
            "day_counter": self.day_counter,
        }
        with open(os.path.join(path, "agent_state.pkl"), "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str) -> "TradingSimulator":
        """Load agent state from checkpoint."""
        with open(os.path.join(path, "agent_state.pkl"), "rb") as f:
            state = pickle.load(f)
        
        brain = BrainDB.load_checkpoint(os.path.join(path, "brain"))
        
        agent = cls(config=state["config"], brain=brain)
        agent.portfolio = state["portfolio"]
        agent.profile = state.get("profile", AgentProfile.create_self_adaptive())
        agent.reflection_results = state["reflection_results"]
        agent.day_counter = state["day_counter"]
        
        return agent
