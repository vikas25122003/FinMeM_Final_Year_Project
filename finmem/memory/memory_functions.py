"""
Memory Scoring Functions

Configurable scoring components for the layered memory system.
Based on the FinMEM paper's memory scoring mechanisms:
- Importance score initialization and changes
- Recency score with exponential decay
- Compound score calculation (importance + recency + similarity)
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ExponentialDecay:
    """Exponential decay function for memory scores.
    
    Decays recency and importance scores each time step (day).
    
    recency_score = e^(-decay_rate * delta)
    importance_score stays the same unless access counter changes it
    delta increments by 1 each step
    """
    
    decay_rate: float = 0.99  # How fast recency decays
    importance_decay_rate: float = 0.0  # Usually 0 — importance doesn't auto-decay
    
    def __call__(
        self,
        important_score: float,
        delta: int
    ) -> Tuple[float, float, int]:
        """Apply decay and return new (recency, importance, delta).
        
        Args:
            important_score: Current importance score.
            delta: Number of time steps since memory was created/promoted.
            
        Returns:
            Tuple of (new_recency, new_importance, new_delta).
        """
        new_delta = delta + 1
        new_recency = math.exp(-self.decay_rate * new_delta)
        new_importance = important_score * (1.0 - self.importance_decay_rate)
        return new_recency, new_importance, new_delta


@dataclass
class LinearCompoundScore:
    """Linear compound score: weighted sum of recency, importance, similarity.
    
    compound = w_recency * recency + w_importance * importance
    final = w_compound * compound + w_similarity * similarity
    """
    
    w_recency: float = 0.5
    w_importance: float = 0.5
    w_similarity: float = 0.5
    w_compound: float = 0.5
    
    def recency_and_importance_score(
        self,
        recency_score: float,
        importance_score: float
    ) -> float:
        """Calculate partial compound score (without similarity)."""
        return self.w_recency * recency_score + self.w_importance * importance_score
    
    def merge_score(
        self,
        similarity_score: float,
        partial_compound_score: float
    ) -> float:
        """Merge similarity with the partial compound score for final ranking."""
        return (
            self.w_similarity * similarity_score
            + self.w_compound * partial_compound_score
        )


@dataclass
class ImportanceScoreInitialization:
    """Initialize importance scores for new memories.
    
    Different layers may start with different base importance.
    """
    
    base_score: float = 0.5
    
    def __call__(self) -> float:
        """Return the initial importance score."""
        return self.base_score


def get_importance_score_initialization(
    memory_layer: str,
    init_type: str = "constant"
) -> ImportanceScoreInitialization:
    """Factory to create importance initialization functions per layer.
    
    Args:
        memory_layer: Layer name (short, mid, long, reflection).
        init_type: Initialization type.
        
    Returns:
        ImportanceScoreInitialization instance.
    """
    # Default: short=0.3, mid=0.5, long=0.7, reflection=0.6
    default_scores = {
        "short": 0.3,
        "mid": 0.5,
        "long": 0.7,
        "reflection": 0.6
    }
    base = default_scores.get(memory_layer, 0.5)
    return ImportanceScoreInitialization(base_score=base)


@dataclass
class RecencyScoreInitialization:
    """Initialize recency score for new memories (always starts at 1.0)."""
    
    base_score: float = 1.0
    
    def __call__(self) -> float:
        """Return the initial recency score."""
        return self.base_score


@dataclass
class LinearImportanceScoreChange:
    """Change importance based on access counter feedback.
    
    When a memory contributes to a profitable trade, its importance increases.
    When it contributes to a loss, importance decreases.
    """
    
    positive_change: float = 0.1   # Increase per positive feedback
    negative_change: float = -0.05  # Decrease per negative feedback
    min_score: float = 0.0
    max_score: float = 1.0
    
    def __call__(
        self,
        access_counter: int,
        importance_score: float
    ) -> float:
        """Update importance based on access counter.
        
        Args:
            access_counter: Cumulative feedback (+1 for profit, -1 for loss).
            importance_score: Current importance score.
            
        Returns:
            Updated importance score.
        """
        if access_counter > 0:
            new_score = importance_score + self.positive_change
        elif access_counter < 0:
            new_score = importance_score + self.negative_change
        else:
            new_score = importance_score
        
        return max(self.min_score, min(self.max_score, new_score))
