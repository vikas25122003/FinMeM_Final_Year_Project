"""
Memory Scoring Functions

Configurable scoring components for the layered memory system.
Based on the FinMEM paper's memory scoring mechanisms:
- Importance score initialization (piecewise probabilistic, per-layer)
- Recency score with exponential decay: S_Recency = e^(-δ / Q_l)
- Importance decay: S_Importance = v * α_l^δ
- Compound score: γ = S_Recency + S_Relevancy + S_Importance (additive sum)

Objective 1 Extension:
- AdaptiveExponentialDecay: Replaces fixed Q_l with regime-conditioned Q_l(regime)
  Enable by setting env var  ADAPTIVE_Q=true
"""

import math
import os
import random
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


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
        delta: int,
        **kwargs,   # absorbs ticker/current_date passed by AdaptiveExponentialDecay callers
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

    Objective 2 Extension:
        When LEARNED_IMPORTANCE=true, uses a trained classifier to predict v_E
        instead of random sampling. Maps the prediction from [40,80] → [0.4,0.8].
    """
    
    values: List[float] = field(default_factory=lambda: [0.4, 0.6, 0.8])
    probabilities: List[float] = field(default_factory=lambda: [0.33, 0.34, 0.33])
    layer_name: str = "short"  # Set by get_importance_score_initialization()
    
    def __call__(self, text_length: int = 100, sentiment_score: float = 0.0) -> float:
        """Return an importance score — learned or random.
        
        When LEARNED_IMPORTANCE=true, delegates to Objective 2 inference.
        Otherwise, uses the paper's random sampling.
        
        Args:
            text_length:     Character count of the memory text (Obj2 feature).
            sentiment_score: Sentiment score in [-1, 1] (Obj2 feature).
        """
        if os.environ.get("LEARNED_IMPORTANCE", "false").lower() == "true":
            try:
                from agentic.obj2_importance.inference import get_importance_score
                v_E = get_importance_score(
                    layer=self.layer_name,
                    age_days=0,
                    access_count=0,
                    text_length=text_length,
                    sentiment_score=sentiment_score,
                )
                # Map from [40, 80] → [0.4, 0.8]
                return v_E / 100.0
            except Exception as exc:
                logger.debug(f"[Obj2] Learned importance failed, using random: {exc}")
        
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
        probabilities=probs,
        layer_name=memory_layer,
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


# ─── Objective 1: Regime-Conditioned Adaptive Decay ────────────────────────

@dataclass
class AdaptiveExponentialDecay:
    """
    Regime-conditioned drop-in replacement for ExponentialDecay.

    Objective 1 of Agentic FinMEM.

    Behaviour:
        ADAPTIVE_Q=false (or unset):
            → Identical to ExponentialDecay (paper defaults, no network calls)
        ADAPTIVE_Q=true:
            → Calls yfinance to detect market regime (CRISIS/SIDEWAYS/BULL)
            → Uses Q from Q_TABLE[regime][layer] instead of the fixed decay_rate
            → Falls back to fixed decay_rate on ANY exception (network, data, etc.)

    Args:
        layer_name:       Memory layer name — "short" | "mid" | "long" | "reflection"
        decay_rate:       Paper's original Q for this layer (14 / 90 / 365)
        importance_base:  Paper's α for importance decay (0.9 / 0.967 / 0.988)

    Usage in BrainDB.create_default():
        decay_function = AdaptiveExponentialDecay(
            layer_name="short", decay_rate=14.0, importance_base=0.9
        )
        # Pass ticker + current_date when calling:
        new_recency, new_importance, new_delta = decay_function(
            important_score=0.6, delta=0,
            ticker="TSLA", current_date="2022-10-25"
        )
    """

    layer_name:      str   = "short"
    decay_rate:      float = 14.0   # Paper default Q — used as fallback
    importance_base: float = 0.9    # Paper default α

    # Runtime state (not serialised as part of the config, set per-call)
    _current_ticker: Optional[str] = field(default=None, init=False, repr=False, compare=False)
    _current_date:   Optional[str] = field(default=None, init=False, repr=False, compare=False)

    def _is_adaptive_enabled(self) -> bool:
        """Check ADAPTIVE_Q env var (true/1/yes → enabled)."""
        val = os.environ.get("ADAPTIVE_Q", "false").strip().lower()
        return val in ("true", "1", "yes")

    def _get_adaptive_Q(self, ticker: str, date_str: str) -> float:
        """
        Compute regime-conditioned Q for the current layer.

        Imports are done lazily (only when ADAPTIVE_Q=true) so that
        the base FinMEM system never imports yfinance on startup.

        Returns:
            Q float. Falls back to self.decay_rate on any error.
        """
        try:
            from agentic.obj1_regime.features   import compute_features
            from agentic.obj1_regime.classifier import get_classifier
            from agentic.obj1_regime.q_table    import get_Q

            clf_mode = os.environ.get("ADAPTIVE_Q_MODE", "threshold").lower()
            features = compute_features(ticker, date_str)
            regime   = get_classifier(mode=clf_mode).predict(features)
            q_val    = float(get_Q(regime, self.layer_name))

            logger.debug(
                f"[AdaptiveQ] layer={self.layer_name} ticker={ticker} "
                f"date={date_str} regime={regime} Q={q_val}"
            )
            return q_val

        except Exception as exc:
            logger.debug(
                f"[AdaptiveQ] Falling back to default Q={self.decay_rate} "
                f"for layer={self.layer_name}: {exc}"
            )
            return self.decay_rate

    def __call__(
        self,
        important_score: float,
        delta: int,
        ticker: Optional[str] = None,
        current_date: Optional[str] = None,
    ) -> Tuple[float, float, int]:
        """
        Apply adaptive (or fixed) exponential decay.

        Args:
            important_score: Current importance score.
            delta:           Days since memory creation/promotion.
            ticker:          Stock ticker for regime detection (optional).
            current_date:    Date string YYYY-MM-DD for regime detection (optional).

        Returns:
            Tuple (new_recency, new_importance, new_delta).
        """
        new_delta = delta + 1

        # Determine Q value
        if (
            self._is_adaptive_enabled()
            and ticker
            and current_date
        ):
            q = self._get_adaptive_Q(ticker, current_date)
        else:
            q = self.decay_rate

        new_recency    = math.exp(-new_delta / q)
        new_importance = important_score * (self.importance_base ** new_delta)
        return new_recency, new_importance, new_delta
