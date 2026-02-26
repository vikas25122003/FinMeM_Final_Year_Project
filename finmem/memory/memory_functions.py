"""
Memory Scoring Functions

Configurable scoring components for the layered memory system.
Based on the FinMEM paper's memory scoring mechanisms:
- Importance score initialization (piecewise probabilistic, per-layer)
- Recency score with exponential decay: S_Recency = e^(-δ / Q_l)
- Importance decay: S_Importance = v * α_l^δ
- Compound score: γ = S_Recency + S_Relevancy + S_Importance (additive sum)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class ExponentialDecay:
    """Exponential decay function for memory scores.
    
    Paper formula:
        S_Recency = e^(-δ / Q_l)
    
    Where:
        δ = number of time steps (days) since memory creation/promotion
        Q_l = stability term (characteristic decay time in days)
            Q_shallow       = 14 days
            Q_intermediate  = 90 days
            Q_deep          = 365 days
    
    Importance decay:
        S_Importance = importance * α_l^δ
    Where:
        α_shallow       = 0.9
        α_intermediate  = 0.967
        α_deep          = 0.988
    """
    
    decay_rate: float = 14.0  # Q_l: stability term in days (higher = slower decay)
    importance_base: float = 0.9  # α_l: importance decay base (higher = slower decay)
    
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
        # Paper: S_Recency = e^(-δ / Q_l)
        new_recency = math.exp(-new_delta / self.decay_rate)
        # Paper: S_Importance = v × α_l^δ
        new_importance = important_score * (self.importance_base ** new_delta)
        return new_recency, new_importance, new_delta


@dataclass
class LinearCompoundScore:
    """Compound score: additive sum of recency, importance, similarity.
    
    Paper formula:
        γ(E) = S_Recency(E) + S_Relevancy(E) + S_Importance(E)
    
    All terms normalized to [0, 1] before summation.
    Final score used for memory retrieval ranking.
    """
    
    # Weights kept for backward compat but default to 1.0 (paper uses equal additive sum)
    w_recency: float = 1.0
    w_importance: float = 1.0
    w_similarity: float = 1.0
    w_compound: float = 1.0
    
    def recency_and_importance_score(
        self,
        recency_score: float,
        importance_score: float
    ) -> float:
        """Calculate partial compound score (without similarity).
        
        Paper: γ_partial = S_Recency + S_Importance
        Both should be in [0, 1].
        """
        # Clamp to [0, 1] per paper
        r = max(0.0, min(1.0, recency_score))
        i = max(0.0, min(1.0, importance_score))
        return self.w_recency * r + self.w_importance * i
    
    def merge_score(
        self,
        similarity_score: float,
        partial_compound_score: float
    ) -> float:
        """Merge similarity with the partial compound score for final ranking.
        
        Paper: γ = S_Recency + S_Relevancy + S_Importance
        Here: final = similarity + partial_compound (which is recency + importance)
        """
        s = max(0.0, min(1.0, similarity_score))
        return self.w_similarity * s + self.w_compound * partial_compound_score


@dataclass
class ImportanceScoreInitialization:
    """Initialize importance scores for new memories.
    
    Paper uses piecewise probabilistic initialization:
        v ∈ {0.4, 0.6, 0.8} with layer-dependent probabilities.
    
    Shallow:       P(0.4)=0.6, P(0.6)=0.3, P(0.8)=0.1
    Intermediate:  P(0.4)=0.2, P(0.6)=0.6, P(0.8)=0.2
    Deep:          P(0.4)=0.1, P(0.6)=0.3, P(0.8)=0.6
    """
    
    values: List[float] = field(default_factory=lambda: [0.4, 0.6, 0.8])
    probabilities: List[float] = field(default_factory=lambda: [0.33, 0.34, 0.33])
    
    def __call__(self) -> float:
        """Return a probabilistically sampled importance score."""
        return random.choices(self.values, weights=self.probabilities, k=1)[0]


def get_importance_score_initialization(
    memory_layer: str,
    init_type: str = "probabilistic"
) -> ImportanceScoreInitialization:
    """Factory to create importance initialization functions per layer.
    
    Paper's piecewise probability assignments:
        Shallow:       high prob of 0.4 (daily news, lower baseline importance)
        Intermediate:  high prob of 0.6 (quarterly filings, medium importance)
        Deep:          high prob of 0.8 (annual filings, high baseline importance)
    
    Args:
        memory_layer: Layer name (short, mid, long, reflection).
        init_type: Initialization type.
        
    Returns:
        ImportanceScoreInitialization instance.
    """
    layer_probs = {
        "short":      [0.6, 0.3, 0.1],   # Bias toward low importance
        "mid":        [0.2, 0.6, 0.2],   # Bias toward medium importance
        "long":       [0.1, 0.3, 0.6],   # Bias toward high importance
        "reflection": [0.1, 0.3, 0.6],   # Same as deep layer
    }
    probs = layer_probs.get(memory_layer, [0.33, 0.34, 0.33])
    return ImportanceScoreInitialization(
        values=[0.4, 0.6, 0.8],
        probabilities=probs
    )


@dataclass
class RecencyScoreInitialization:
    """Initialize recency score for new memories (always starts at 1.0).
    
    Paper: S_Recency starts at 1.0 (fresh memory, δ=0).
    """
    
    base_score: float = 1.0
    
    def __call__(self) -> float:
        """Return the initial recency score."""
        return self.base_score


@dataclass
class LinearImportanceScoreChange:
    """Change importance based on access counter feedback.
    
    Paper: When a memory is identified as "pivotal for investment success"
    by the LLM reflection, it receives a bonus to its importance score.
    
    Additionally, standard feedback:
    - Profitable trade: +1 to access counter → importance increase
    - Loss trade: -1 to access counter → importance decrease
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
