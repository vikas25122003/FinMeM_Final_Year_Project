"""
Layered Memory System

The core innovation of FinMEM — a 4-layer memory system (short/mid/long/reflection)
backed by FAISS for vector search, with:
- Per-layer exponential decay
- Memory promotion/demotion (jump mechanism)
- Access counter feedback from trading outcomes
- Compound scoring: recency * importance * similarity

Based on the FinMEM paper and reference implementation.
"""

import os
import pickle
import logging
import math
import numpy as np
from datetime import date, datetime
from typing import List, Optional, Dict, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from .embeddings import get_embedding_model, EmbeddingModel
from .memory_functions import (
    ExponentialDecay,
    LinearCompoundScore,
    ImportanceScoreInitialization,
    RecencyScoreInitialization,
    LinearImportanceScoreChange,
    get_importance_score_initialization,
)

logger = logging.getLogger(__name__)


class MemoryLayer(Enum):
    """Memory layer types."""
    SHORT = "short"
    MID = "mid"
    LONG = "long"
    REFLECTION = "reflection"


# ─── Low-level MemoryDB (one per layer) ────────────────────────────────────

class _IdGenerator:
    """Thread-safe incrementing ID generator."""
    def __init__(self):
        self.current_id = 0

    def __call__(self) -> int:
        self.current_id += 1
        return self.current_id - 1


class MemoryDB:
    """Single-layer memory database backed by numpy for vector search.
    
    Each layer has its own MemoryDB with configurable:
    - Jump thresholds for promotion/demotion
    - Decay parameters
    - Cleanup thresholds
    - Importance initialization
    """

    def __init__(
        self,
        db_name: str,
        id_generator: _IdGenerator,
        emb_dim: int,
        jump_threshold_upper: float,
        jump_threshold_lower: float,
        importance_score_init: ImportanceScoreInitialization,
        recency_score_init: RecencyScoreInitialization,
        compound_score_calc: LinearCompoundScore,
        importance_score_change: LinearImportanceScoreChange,
        decay_function: ExponentialDecay,
        clean_up_threshold_dict: Dict[str, float],
    ):
        self.db_name = db_name
        self.id_generator = id_generator
        self.jump_threshold_upper = jump_threshold_upper
        self.jump_threshold_lower = jump_threshold_lower
        self.emb_dim = emb_dim
        self.importance_score_init = importance_score_init
        self.recency_score_init = recency_score_init
        self.compound_score_calc = compound_score_calc
        self.importance_score_change = importance_score_change
        self.decay_function = decay_function
        self.clean_up_threshold_dict = dict(clean_up_threshold_dict)

        # Per-symbol storage
        # universe[symbol] = {
        #   "score_memory": [ {text, id, important_score, recency_score, delta,
        #                      compound_score, access_counter, date, embedding}, ... ]
        #   "embeddings": np.ndarray  (N x emb_dim)
        #   "ids": list[int]
        # }
        self.universe: Dict[str, Dict[str, Any]] = {}

    def _ensure_symbol(self, symbol: str) -> None:
        """Create storage for a symbol if it doesn't exist."""
        if symbol not in self.universe:
            self.universe[symbol] = {
                "score_memory": [],
                "embeddings": np.empty((0, self.emb_dim), dtype=np.float32),
                "ids": [],
            }

    def add_memory(
        self,
        symbol: str,
        mem_date: Union[date, datetime],
        text: Union[List[str], str],
        embeddings: np.ndarray,
    ) -> List[int]:
        """Add one or more memories.
        
        Args:
            symbol: Stock ticker.
            mem_date: Date of the memory.
            text: Text content(s).
            embeddings: Pre-computed embeddings (N x emb_dim).
            
        Returns:
            List of assigned memory IDs.
        """
        self._ensure_symbol(symbol)

        if isinstance(text, str):
            text = [text]
            embeddings = embeddings.reshape(1, -1)

        # Normalize embeddings for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        ids = []
        for i, t in enumerate(text):
            mem_id = self.id_generator()
            importance = self.importance_score_init()
            recency = self.recency_score_init()
            compound = self.compound_score_calc.recency_and_importance_score(
                recency, importance
            )

            record = {
                "text": t,
                "id": mem_id,
                "important_score": importance,
                "recency_score": recency,
                "delta": 0,
                "compound_score": compound,
                "access_counter": 0,
                "date": mem_date,
            }
            self.universe[symbol]["score_memory"].append(record)
            
            # Add embedding
            emb = embeddings[i:i+1].astype(np.float32)
            self.universe[symbol]["embeddings"] = np.vstack(
                [self.universe[symbol]["embeddings"], emb]
            )
            self.universe[symbol]["ids"].append(mem_id)
            ids.append(mem_id)

        return ids

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        symbol: str,
    ) -> Tuple[List[str], List[int]]:
        """Query memories by combining similarity + compound score.
        
        Two-phase search (matching reference implementation):
        1. Find top-k by cosine similarity, merge with compound scores
        2. Find top-k by compound score, get their similarity
        3. Rank all candidates by merged score, return unique top-k
        
        Args:
            query_embedding: Query vector (1 x emb_dim).
            top_k: Number of results.
            symbol: Stock ticker.
            
        Returns:
            Tuple of (text_list, id_list).
        """
        if (
            symbol not in self.universe
            or len(self.universe[symbol]["score_memory"]) == 0
            or top_k == 0
        ):
            return [], []

        records = self.universe[symbol]["score_memory"]
        all_embeddings = self.universe[symbol]["embeddings"]
        all_ids = self.universe[symbol]["ids"]
        n = len(records)
        top_k = min(top_k, n)

        # Normalize query
        query_emb = query_embedding.reshape(1, -1).astype(np.float32)
        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm

        # Phase 1: top-k by cosine similarity
        similarities = (all_embeddings @ query_emb.T).flatten()
        p1_indices = np.argsort(similarities)[::-1][:top_k]

        # Phase 2: top-k by compound score
        compound_scores = [r["compound_score"] for r in records]
        p2_indices = np.argsort(compound_scores)[::-1][:top_k]

        # Merge candidates
        candidate_set = set(p1_indices.tolist()) | set(p2_indices.tolist())
        
        candidates = []
        for idx in candidate_set:
            sim = float(similarities[idx])
            compound = records[idx]["compound_score"]
            final = self.compound_score_calc.merge_score(sim, compound)
            candidates.append((idx, final))

        # Sort by final score, take top-k unique
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        ret_texts = []
        ret_ids = []
        seen = set()
        for idx, _ in candidates:
            mid = all_ids[idx]
            if mid not in seen:
                seen.add(mid)
                ret_texts.append(records[idx]["text"])
                ret_ids.append(mid)
            if len(ret_ids) >= top_k:
                break

        return ret_texts, ret_ids

    def update_access_count_with_feedback(
        self,
        symbol: str,
        ids: List[int],
        feedbacks: List[int],
    ) -> List[int]:
        """Update access counters and importance scores based on feedback.
        
        Args:
            symbol: Stock ticker.
            ids: Memory IDs to update.
            feedbacks: Feedback values (+1 for profit, -1 for loss).
            
        Returns:
            List of successfully updated IDs.
        """
        if symbol not in self.universe:
            return []

        records = self.universe[symbol]["score_memory"]
        success_ids = []
        
        for mem_id, feedback in zip(ids, feedbacks):
            for record in records:
                if record["id"] == mem_id:
                    record["access_counter"] += feedback
                    record["important_score"] = self.importance_score_change(
                        access_counter=record["access_counter"],
                        importance_score=record["important_score"],
                    )
                    record["compound_score"] = (
                        self.compound_score_calc.recency_and_importance_score(
                            recency_score=record["recency_score"],
                            importance_score=record["important_score"],
                        )
                    )
                    success_ids.append(mem_id)
                    break

        return success_ids

    def _decay(self) -> None:
        """Apply exponential decay to all memories."""
        for symbol in self.universe:
            for record in self.universe[symbol]["score_memory"]:
                (
                    record["recency_score"],
                    record["important_score"],
                    record["delta"],
                ) = self.decay_function(
                    important_score=record["important_score"],
                    delta=record["delta"],
                )
                record["compound_score"] = (
                    self.compound_score_calc.recency_and_importance_score(
                        recency_score=record["recency_score"],
                        importance_score=record["important_score"],
                    )
                )

    def _clean_up(self) -> List[int]:
        """Remove memories below score thresholds.
        
        Returns:
            List of removed memory IDs.
        """
        removed_ids = []
        recency_thresh = self.clean_up_threshold_dict.get("recency_threshold", 0.01)
        importance_thresh = self.clean_up_threshold_dict.get("importance_threshold", 0.01)

        for symbol in self.universe:
            records = self.universe[symbol]["score_memory"]
            ids_list = self.universe[symbol]["ids"]
            
            # Find indices to remove
            remove_indices = []
            for i, record in enumerate(records):
                if (record["recency_score"] < recency_thresh or 
                    record["important_score"] < importance_thresh):
                    remove_indices.append(i)
                    removed_ids.append(record["id"])

            if remove_indices:
                # Remove from records, embeddings, and ids
                keep_indices = [i for i in range(len(records)) if i not in remove_indices]
                self.universe[symbol]["score_memory"] = [records[i] for i in keep_indices]
                if len(keep_indices) > 0:
                    self.universe[symbol]["embeddings"] = self.universe[symbol]["embeddings"][keep_indices]
                else:
                    self.universe[symbol]["embeddings"] = np.empty((0, self.emb_dim), dtype=np.float32)
                self.universe[symbol]["ids"] = [ids_list[i] for i in keep_indices]

        return removed_ids

    def step(self) -> List[int]:
        """One time step: decay then clean up.
        
        Returns:
            List of removed memory IDs.
        """
        self._decay()
        return self._clean_up()

    def prepare_jump(self) -> Tuple[Dict, Dict, List[int]]:
        """Prepare memories for promotion/demotion.
        
        Memories with importance >= jump_threshold_upper are promoted (jump up).
        Memories with importance < jump_threshold_lower are demoted (jump down).
        
        Returns:
            (jump_up_dict, jump_down_dict, removed_ids)
            Each dict: {symbol: {"objects": [...], "embeddings": np.ndarray}}
        """
        jump_up = {}
        jump_down = {}
        removed_ids = []

        for symbol in self.universe:
            records = self.universe[symbol]["score_memory"]
            ids_list = self.universe[symbol]["ids"]
            embeddings = self.universe[symbol]["embeddings"]
            
            up_indices = []
            down_indices = []

            for i, record in enumerate(records):
                if record["important_score"] >= self.jump_threshold_upper:
                    up_indices.append(i)
                elif record["important_score"] < self.jump_threshold_lower:
                    down_indices.append(i)

            all_remove = up_indices + down_indices
            for idx in all_remove:
                removed_ids.append(records[idx]["id"])

            if up_indices:
                jump_up[symbol] = {
                    "objects": [records[i] for i in up_indices],
                    "embeddings": embeddings[up_indices],
                }

            if down_indices:
                jump_down[symbol] = {
                    "objects": [records[i] for i in down_indices],
                    "embeddings": embeddings[down_indices],
                }

            # Remove jumped memories from this layer
            if all_remove:
                keep = [i for i in range(len(records)) if i not in all_remove]
                self.universe[symbol]["score_memory"] = [records[i] for i in keep]
                if len(keep) > 0:
                    self.universe[symbol]["embeddings"] = embeddings[keep]
                else:
                    self.universe[symbol]["embeddings"] = np.empty((0, self.emb_dim), dtype=np.float32)
                self.universe[symbol]["ids"] = [ids_list[i] for i in keep]

        return jump_up, jump_down, removed_ids

    def accept_jump(self, jump_dict: Dict, direction: str) -> None:
        """Accept memories jumping in from another layer.
        
        Args:
            jump_dict: {symbol: {"objects": [...], "embeddings": np.ndarray}}
            direction: "up" (promoted) or "down" (demoted).
        """
        for symbol, data in jump_dict.items():
            self._ensure_symbol(symbol)

            for i, obj in enumerate(data["objects"]):
                if direction == "up":
                    # Reset recency on promotion
                    obj["recency_score"] = self.recency_score_init()
                    obj["delta"] = 0
                
                obj["compound_score"] = (
                    self.compound_score_calc.recency_and_importance_score(
                        recency_score=obj["recency_score"],
                        importance_score=obj["important_score"],
                    )
                )
                self.universe[symbol]["score_memory"].append(obj)
                self.universe[symbol]["ids"].append(obj["id"])

            embs = data["embeddings"]
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            self.universe[symbol]["embeddings"] = np.vstack(
                [self.universe[symbol]["embeddings"], embs.astype(np.float32)]
            )

    def get_memory_count(self, symbol: str) -> int:
        """Get number of memories for a symbol."""
        if symbol not in self.universe:
            return 0
        return len(self.universe[symbol]["score_memory"])

    def save_checkpoint(self, name: str, path: str) -> None:
        """Save this memory layer to disk."""
        layer_path = os.path.join(path, name)
        os.makedirs(layer_path, exist_ok=True)

        state = {
            "db_name": self.db_name,
            "jump_threshold_upper": self.jump_threshold_upper,
            "jump_threshold_lower": self.jump_threshold_lower,
            "emb_dim": self.emb_dim,
            "importance_score_init": self.importance_score_init,
            "recency_score_init": self.recency_score_init,
            "compound_score_calc": self.compound_score_calc,
            "importance_score_change": self.importance_score_change,
            "decay_function": self.decay_function,
            "clean_up_threshold_dict": self.clean_up_threshold_dict,
        }
        with open(os.path.join(layer_path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state, f)
        
        # Save universe (memories + embeddings)
        save_data = {}
        for symbol in self.universe:
            save_data[symbol] = {
                "score_memory": self.universe[symbol]["score_memory"],
                "embeddings": self.universe[symbol]["embeddings"],
                "ids": self.universe[symbol]["ids"],
            }
        with open(os.path.join(layer_path, "universe.pkl"), "wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def load_checkpoint(cls, name: str, path: str, id_generator: _IdGenerator) -> "MemoryDB":
        """Load a memory layer from disk."""
        layer_path = os.path.join(path, name)
        
        with open(os.path.join(layer_path, "state_dict.pkl"), "rb") as f:
            state = pickle.load(f)
        with open(os.path.join(layer_path, "universe.pkl"), "rb") as f:
            universe_data = pickle.load(f)

        obj = cls(
            db_name=state["db_name"],
            id_generator=id_generator,
            emb_dim=state["emb_dim"],
            jump_threshold_upper=state["jump_threshold_upper"],
            jump_threshold_lower=state["jump_threshold_lower"],
            importance_score_init=state["importance_score_init"],
            recency_score_init=state["recency_score_init"],
            compound_score_calc=state["compound_score_calc"],
            importance_score_change=state["importance_score_change"],
            decay_function=state["decay_function"],
            clean_up_threshold_dict=state["clean_up_threshold_dict"],
        )
        obj.universe = universe_data
        return obj


# ─── BrainDB: 4-layer orchestrator ────────────────────────────────────────

class BrainDB:
    """Four-layer memory system orchestrating short/mid/long/reflection MemoryDBs.
    
    Handles:
    - Adding memories to the correct layer
    - Querying across layers
    - Memory jumps (promotion/demotion) between layers
    - Decay and cleanup across all layers
    """

    def __init__(
        self,
        agent_name: str,
        emb_model: EmbeddingModel,
        id_generator: _IdGenerator,
        short_term_memory: MemoryDB,
        mid_term_memory: MemoryDB,
        long_term_memory: MemoryDB,
        reflection_memory: MemoryDB,
    ):
        self.agent_name = agent_name
        self.emb_model = emb_model
        self.id_generator = id_generator
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.reflection_memory = reflection_memory
        self.removed_ids: List[int] = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BrainDB":
        """Create BrainDB from a config dictionary.
        
        Args:
            config: Must contain keys: agent_name, top_k, and per-layer
                    settings under 'short', 'mid', 'long', 'reflection'.
        """
        from ..config import DEFAULT_CONFIG
        
        id_gen = _IdGenerator()
        emb_model = get_embedding_model()
        emb_dim = len(emb_model.embed("test"))
        agent_name = config.get("agent_name", "finmem_agent")

        # Helper to build a MemoryDB from layer config
        def _build_layer(name: str, layer_cfg: Dict) -> MemoryDB:
            return MemoryDB(
                db_name=f"{agent_name}_{name}",
                id_generator=id_gen,
                emb_dim=emb_dim,
                jump_threshold_upper=layer_cfg.get("jump_threshold_upper", 999999),
                jump_threshold_lower=layer_cfg.get("jump_threshold_lower", -999999),
                importance_score_init=get_importance_score_initialization(name),
                recency_score_init=RecencyScoreInitialization(),
                compound_score_calc=LinearCompoundScore(
                    **layer_cfg.get("compound_score_params", {})
                ),
                importance_score_change=LinearImportanceScoreChange(),
                decay_function=ExponentialDecay(
                    **layer_cfg.get("decay_params", {})
                ),
                clean_up_threshold_dict=layer_cfg.get(
                    "clean_up_threshold_dict",
                    {"recency_threshold": 0.01, "importance_threshold": 0.01}
                ),
            )

        short_cfg = config.get("short", {})
        mid_cfg = config.get("mid", {})
        long_cfg = config.get("long", {})
        reflection_cfg = config.get("reflection", {})

        return cls(
            agent_name=agent_name,
            emb_model=emb_model,
            id_generator=id_gen,
            short_term_memory=_build_layer("short", short_cfg),
            mid_term_memory=_build_layer("mid", mid_cfg),
            long_term_memory=_build_layer("long", long_cfg),
            reflection_memory=_build_layer("reflection", reflection_cfg),
        )

    @classmethod
    def create_default(cls) -> "BrainDB":
        """Create a BrainDB with sensible defaults."""
        return cls.from_config({
            "agent_name": "finmem_agent",
            "short": {
                "jump_threshold_upper": 0.8,
                "jump_threshold_lower": -999999,  # No demotion from short
                "decay_params": {"decay_rate": 0.99},
                "clean_up_threshold_dict": {"recency_threshold": 0.01, "importance_threshold": 0.01},
            },
            "mid": {
                "jump_threshold_upper": 0.85,
                "jump_threshold_lower": 0.1,
                "decay_params": {"decay_rate": 0.5},
                "clean_up_threshold_dict": {"recency_threshold": 0.01, "importance_threshold": 0.01},
            },
            "long": {
                "jump_threshold_upper": 999999,  # No promotion from long
                "jump_threshold_lower": 0.15,
                "decay_params": {"decay_rate": 0.1},
                "clean_up_threshold_dict": {"recency_threshold": 0.005, "importance_threshold": 0.005},
            },
            "reflection": {
                "jump_threshold_upper": 999999,  # No promotion
                "jump_threshold_lower": -999999,  # No demotion
                "decay_params": {"decay_rate": 0.3},
                "clean_up_threshold_dict": {"recency_threshold": 0.005, "importance_threshold": 0.005},
            },
        })

    def _embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        if isinstance(text, str):
            return np.array(self.emb_model.embed(text), dtype=np.float32).reshape(1, -1)
        else:
            return np.array(self.emb_model.embed_batch(text), dtype=np.float32)

    # ── Add methods ──

    def add_memory_short(self, symbol: str, mem_date: Union[date, datetime], text: Union[str, List[str]]) -> List[int]:
        """Add news/short-term data to short memory."""
        embs = self._embed(text)
        return self.short_term_memory.add_memory(symbol, mem_date, text, embs)

    def add_memory_mid(self, symbol: str, mem_date: Union[date, datetime], text: Union[str, List[str]]) -> List[int]:
        """Add quarterly filings/trends to mid memory."""
        embs = self._embed(text)
        return self.mid_term_memory.add_memory(symbol, mem_date, text, embs)

    def add_memory_long(self, symbol: str, mem_date: Union[date, datetime], text: Union[str, List[str]]) -> List[int]:
        """Add annual filings/fundamentals to long memory."""
        embs = self._embed(text)
        return self.long_term_memory.add_memory(symbol, mem_date, text, embs)

    def add_memory_reflection(self, symbol: str, mem_date: Union[date, datetime], text: Union[str, List[str]]) -> List[int]:
        """Add reflection summaries to reflection memory."""
        embs = self._embed(text)
        return self.reflection_memory.add_memory(symbol, mem_date, text, embs)

    # ── Query methods ──

    def query_short(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        """Query short-term memory."""
        emb = self._embed(query_text)
        return self.short_term_memory.query(emb, top_k, symbol)

    def query_mid(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        """Query mid-term memory."""
        emb = self._embed(query_text)
        return self.mid_term_memory.query(emb, top_k, symbol)

    def query_long(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        """Query long-term memory."""
        emb = self._embed(query_text)
        return self.long_term_memory.query(emb, top_k, symbol)

    def query_reflection(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        """Query reflection memory."""
        emb = self._embed(query_text)
        return self.reflection_memory.query(emb, top_k, symbol)

    # ── Access counter feedback ──

    def update_access_count_with_feedback(
        self,
        symbol: str,
        ids: Union[List[int], int],
        feedback: int,
    ) -> None:
        """Update access counters across all layers.
        
        Searches each layer for matching IDs to update.
        
        Args:
            symbol: Stock ticker.
            ids: Memory ID(s) to update.
            feedback: +1 for profitable trade, -1 for loss, 0 for neutral.
        """
        if isinstance(ids, int):
            ids = [ids]
        
        # Skip removed IDs
        ids = [i for i in ids if i not in self.removed_ids]
        if not ids:
            return

        feedbacks = [feedback] * len(ids)
        
        # Try each layer
        success = self.short_term_memory.update_access_count_with_feedback(
            symbol, ids, feedbacks
        )
        remaining = [i for i in ids if i not in success]
        if not remaining:
            return

        feedbacks = [feedback] * len(remaining)
        success += self.mid_term_memory.update_access_count_with_feedback(
            symbol, remaining, feedbacks
        )
        remaining = [i for i in remaining if i not in success]
        if not remaining:
            return

        feedbacks = [feedback] * len(remaining)
        success += self.long_term_memory.update_access_count_with_feedback(
            symbol, remaining, feedbacks
        )
        remaining = [i for i in remaining if i not in success]
        if not remaining:
            return

        feedbacks = [feedback] * len(remaining)
        self.reflection_memory.update_access_count_with_feedback(
            symbol, remaining, feedbacks
        )

    # ── Step: decay, cleanup, jump ──

    def step(self) -> None:
        """One time step across all layers.
        
        1. Decay and cleanup each layer
        2. Process memory jumps between layers (2 iterations)
        """
        # Step 1: decay + cleanup
        self.removed_ids.extend(self.short_term_memory.step())
        self.removed_ids.extend(self.mid_term_memory.step())
        self.removed_ids.extend(self.long_term_memory.step())
        self.removed_ids.extend(self.reflection_memory.step())

        # Step 2: memory jumps (run twice for cascading)
        for _ in range(2):
            # Short → Up to Mid
            up, down, deleted = self.short_term_memory.prepare_jump()
            self.removed_ids.extend(deleted)
            if up:
                self.mid_term_memory.accept_jump(up, "up")

            # Mid → Up to Long, Down to Short
            up, down, deleted = self.mid_term_memory.prepare_jump()
            self.removed_ids.extend(deleted)
            if up:
                self.long_term_memory.accept_jump(up, "up")
            if down:
                self.short_term_memory.accept_jump(down, "down")

            # Long → Down to Mid
            up, down, deleted = self.long_term_memory.prepare_jump()
            self.removed_ids.extend(deleted)
            if down:
                self.mid_term_memory.accept_jump(down, "down")

    # ── Stats ──

    def stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics."""
        def _layer_count(layer: MemoryDB) -> int:
            if symbol:
                return layer.get_memory_count(symbol)
            return sum(layer.get_memory_count(s) for s in layer.universe)

        return {
            "short": _layer_count(self.short_term_memory),
            "mid": _layer_count(self.mid_term_memory),
            "long": _layer_count(self.long_term_memory),
            "reflection": _layer_count(self.reflection_memory),
            "total_removed": len(self.removed_ids),
        }

    # ── Checkpointing ──

    def save_checkpoint(self, path: str) -> None:
        """Save entire BrainDB to disk."""
        os.makedirs(path, exist_ok=True)

        state = {
            "agent_name": self.agent_name,
            "removed_ids": self.removed_ids,
            "id_generator": self.id_generator,
        }
        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state, f)

        self.short_term_memory.save_checkpoint("short", path)
        self.mid_term_memory.save_checkpoint("mid", path)
        self.long_term_memory.save_checkpoint("long", path)
        self.reflection_memory.save_checkpoint("reflection", path)

    @classmethod
    def load_checkpoint(cls, path: str) -> "BrainDB":
        """Load BrainDB from disk."""
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state = pickle.load(f)

        id_gen = state["id_generator"]
        emb_model = get_embedding_model()

        return cls(
            agent_name=state["agent_name"],
            emb_model=emb_model,
            id_generator=id_gen,
            short_term_memory=MemoryDB.load_checkpoint("short", path, id_gen),
            mid_term_memory=MemoryDB.load_checkpoint("mid", path, id_gen),
            long_term_memory=MemoryDB.load_checkpoint("long", path, id_gen),
            reflection_memory=MemoryDB.load_checkpoint("reflection", path, id_gen),
        )
