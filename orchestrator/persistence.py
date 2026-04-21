"""
State Persistence — Save and load all system state across restarts.

Handles:
- BrainDB (4-layer memory with embeddings)
- Portfolio state (cash, positions, history)
- TradingAgents BM25 memories
- Trade log history
- Cycle metadata
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class StatePersistence:
    """Manages saving and loading all persistent state."""

    def __init__(self, state_dir: str = "./state"):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        os.makedirs(os.path.join(state_dir, "brain"), exist_ok=True)
        os.makedirs(os.path.join(state_dir, "trades"), exist_ok=True)

    def save_brain(self, brain) -> None:
        """Save BrainDB to disk using its built-in checkpoint mechanism."""
        brain_path = os.path.join(self.state_dir, "brain")
        try:
            brain.short_term_memory.save_checkpoint("short", brain_path)
            brain.mid_term_memory.save_checkpoint("mid", brain_path)
            brain.long_term_memory.save_checkpoint("long", brain_path)
            brain.reflection_memory.save_checkpoint("reflection", brain_path)

            meta = {
                "agent_name": brain.agent_name,
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "layers": {
                    "short": brain.short_term_memory.get_memory_count("__total__") if hasattr(brain.short_term_memory, "get_memory_count") else 0,
                    "mid": brain.mid_term_memory.get_memory_count("__total__") if hasattr(brain.mid_term_memory, "get_memory_count") else 0,
                },
            }
            with open(os.path.join(brain_path, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"[Persistence] BrainDB saved to {brain_path}")
        except Exception as e:
            logger.error(f"[Persistence] Failed to save BrainDB: {e}")

    def load_brain(self):
        """Load BrainDB from disk, or return None if no checkpoint exists."""
        from finmem.memory.layered_memory import BrainDB, MemoryDB, _IdGenerator, get_embedding_model

        brain_path = os.path.join(self.state_dir, "brain")
        short_path = os.path.join(brain_path, "short")

        if not os.path.exists(os.path.join(short_path, "state_dict.pkl")):
            logger.info("[Persistence] No BrainDB checkpoint found — starting fresh")
            return None

        try:
            id_gen = _IdGenerator()
            emb_model = get_embedding_model()

            short = MemoryDB.load_checkpoint("short", brain_path, id_gen)
            mid = MemoryDB.load_checkpoint("mid", brain_path, id_gen)
            long = MemoryDB.load_checkpoint("long", brain_path, id_gen)
            reflection = MemoryDB.load_checkpoint("reflection", brain_path, id_gen)

            brain = BrainDB(
                agent_name="finmem_agent",
                emb_model=emb_model,
                id_generator=id_gen,
                short_term_memory=short,
                mid_term_memory=mid,
                long_term_memory=long,
                reflection_memory=reflection,
            )
            logger.info(f"[Persistence] BrainDB loaded from {brain_path}")
            return brain
        except Exception as e:
            logger.error(f"[Persistence] Failed to load BrainDB: {e}")
            return None

    def save_portfolio_state(self, portfolio_state: Dict[str, Any]) -> None:
        """Save portfolio state (cash, positions, history) to JSON."""
        path = os.path.join(self.state_dir, "portfolio.json")
        try:
            serializable = _make_serializable(portfolio_state)
            with open(path, "w") as f:
                json.dump(serializable, f, indent=2)
            logger.info(f"[Persistence] Portfolio saved to {path}")
        except Exception as e:
            logger.error(f"[Persistence] Failed to save portfolio: {e}")

    def load_portfolio_state(self) -> Optional[Dict[str, Any]]:
        """Load portfolio state from disk."""
        path = os.path.join(self.state_dir, "portfolio.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                state = json.load(f)
            logger.info(f"[Persistence] Portfolio loaded from {path}")
            return state
        except Exception as e:
            logger.error(f"[Persistence] Failed to load portfolio: {e}")
            return None

    def save_ta_memories(self, ta_graph) -> None:
        """Save TradingAgents' BM25 memories to disk."""
        mem_path = os.path.join(self.state_dir, "ta_memories.pkl")
        try:
            memories = {
                "bull": ta_graph.bull_memory,
                "bear": ta_graph.bear_memory,
                "trader": ta_graph.trader_memory,
                "invest_judge": ta_graph.invest_judge_memory,
                "portfolio_manager": ta_graph.portfolio_manager_memory,
            }
            with open(mem_path, "wb") as f:
                pickle.dump(memories, f)
            logger.info(f"[Persistence] TradingAgents memories saved")
        except Exception as e:
            logger.error(f"[Persistence] Failed to save TA memories: {e}")

    def load_ta_memories(self, ta_graph) -> bool:
        """Load TradingAgents' BM25 memories from disk into graph instance."""
        mem_path = os.path.join(self.state_dir, "ta_memories.pkl")
        if not os.path.exists(mem_path):
            return False
        try:
            with open(mem_path, "rb") as f:
                memories = pickle.load(f)
            ta_graph.bull_memory = memories.get("bull", ta_graph.bull_memory)
            ta_graph.bear_memory = memories.get("bear", ta_graph.bear_memory)
            ta_graph.trader_memory = memories.get("trader", ta_graph.trader_memory)
            ta_graph.invest_judge_memory = memories.get("invest_judge", ta_graph.invest_judge_memory)
            ta_graph.portfolio_manager_memory = memories.get("portfolio_manager", ta_graph.portfolio_manager_memory)
            logger.info("[Persistence] TradingAgents memories loaded")
            return True
        except Exception as e:
            logger.error(f"[Persistence] Failed to load TA memories: {e}")
            return False

    def log_trade(self, trade_record: Dict[str, Any]) -> None:
        """Append a trade record to the daily trade log."""
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(self.state_dir, "trades", f"{today}.jsonl")
        try:
            serializable = _make_serializable(trade_record)
            with open(log_path, "a") as f:
                f.write(json.dumps(serializable) + "\n")
        except Exception as e:
            logger.warning(f"[Persistence] Trade log write failed: {e}")

    def save_cycle_state(self, cycle_data: Dict[str, Any]) -> None:
        """Save the latest cycle state for crash recovery."""
        path = os.path.join(self.state_dir, "last_cycle.json")
        try:
            serializable = _make_serializable(cycle_data)
            with open(path, "w") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.warning(f"[Persistence] Cycle state save failed: {e}")

    def load_cycle_state(self) -> Optional[Dict[str, Any]]:
        """Load the last cycle state for crash recovery."""
        path = os.path.join(self.state_dir, "last_cycle.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None


def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-serializable types to strings."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)
