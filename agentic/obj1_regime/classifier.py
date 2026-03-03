"""
Objective 1 — Step 2: Threshold-Based Regime Classifier

Classifies the current market into one of three regimes:
    BULL     — low volatility + positive momentum
    CRISIS   — high volatility (extreme market turbulence)
    SIDEWAYS — everything else (paper default)

Rules (annualized volatility thresholds, grounded in quantitative finance):
    CRISIS   if vol_20d >= 0.35  (annualized)   → ~35% annualized vol
    BULL     if vol_20d <= 0.20 AND momentum_20d >= 0.05
    SIDEWAYS otherwise

Why threshold rules (not HMM/LSTM for now)?
    - Deterministic + reproducible — every decision traceable to a single inequality
    - Standard in quantitative finance (VIX > 30 = fear regime is used professionally)
    - HMM would need labeled regime data and risks lookahead bias in backtesting
    - HMM supported as future work via get_classifier(mode="hmm")

Usage:
    clf    = get_classifier()                      # default: threshold
    regime = clf.predict({"vol_20d": 0.41, ...})  # -> "CRISIS"
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ThresholdRegimeClassifier:
    """
    Rule-based market regime classifier.

    Three regimes:
        CRISIS   — annualized vol >= 35%  (panic, fast-moving markets)
        BULL     — annualized vol <= 20% AND 20d momentum >= +5%
        SIDEWAYS — neutral / undefined (FinMEM paper defaults apply)

    Panel Q&A justification for thresholds:
        0.35 annualized ≈ VIX ~30 (widely used fear gauge threshold)
        0.20 annualized ≈ calm market; news half-life naturally extends
        5% momentum      ≈ stock up >5% in 20 days → sustained trend
    """

    CRISIS_VOL_THRESHOLD    = 0.35   # annualized daily vol (>35% = high fear)
    BULL_VOL_THRESHOLD      = 0.20   # annualized daily vol (<20% = calm)
    BULL_MOMENTUM_THRESHOLD = 0.05   # 20-day price return (>5% = bullish)

    def predict(self, features: Dict[str, float]) -> str:
        """
        Classify market regime from feature dict.

        Args:
            features: Dict with keys vol_20d, vol_50d, momentum_20d.

        Returns:
            Regime string: "CRISIS" | "BULL" | "SIDEWAYS"
        """
        vol_20d      = features.get("vol_20d", 0.20)
        momentum_20d = features.get("momentum_20d", 0.0)

        if vol_20d >= self.CRISIS_VOL_THRESHOLD:
            regime = "CRISIS"
        elif vol_20d <= self.BULL_VOL_THRESHOLD and momentum_20d >= self.BULL_MOMENTUM_THRESHOLD:
            regime = "BULL"
        else:
            regime = "SIDEWAYS"

        logger.debug(
            f"[Obj1] Regime: {regime} | "
            f"vol_20d={vol_20d:.4f} | momentum_20d={momentum_20d:.4f}"
        )
        return regime

    def predict_batch(self, feature_list: list) -> list:
        """Classify a list of feature dicts."""
        return [self.predict(f) for f in feature_list]


def get_classifier(mode: str = "threshold") -> ThresholdRegimeClassifier:
    """
    Factory function to get a regime classifier.

    Args:
        mode: "threshold" (default) or "hmm" (future work).

    Returns:
        ThresholdRegimeClassifier instance.

    Raises:
        NotImplementedError: If mode="hmm" (planned for future).
    """
    if mode == "threshold":
        return ThresholdRegimeClassifier()
    elif mode == "hmm":
        raise NotImplementedError(
            "HMM classifier is future work. "
            "Install hmmlearn and implement HMMRegimeClassifier in classifier.py. "
            "For now use mode='threshold'."
        )
    else:
        raise ValueError(f"Unknown classifier mode: '{mode}'. Use 'threshold' or 'hmm'.")
