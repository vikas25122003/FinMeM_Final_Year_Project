"""
Objective 2 — Step 4: Runtime Importance Inference

Replaces FinMem's random v_E assignment with a learned prediction:
    v_E = 40 + (classifier.predict_proba(features) × 40)

Falls back to the paper's random assignment if no model is loaded.
"""

import os
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level model bundle (loaded once, used everywhere)
_bundle = None


def load_model(path: Optional[str] = None) -> bool:
    """Load the importance classifier from disk.

    Args:
        path: Path to the pkl bundle. Defaults to IMPORTANCE_MODEL_PATH env var.

    Returns:
        True if model loaded successfully, False otherwise.
    """
    global _bundle
    import joblib

    if path is None:
        path = os.getenv("IMPORTANCE_MODEL_PATH", "./models/importance_clf.pkl")

    try:
        _bundle = joblib.load(path)
        logger.info(f"[Obj2] Loaded importance model from {path} "
                     f"(type={_bundle.get('model_type')}, acc={_bundle.get('accuracy', 'N/A')})")
        return True
    except FileNotFoundError:
        logger.info(f"[Obj2] No importance model at {path} — using random fallback")
        return False
    except Exception as exc:
        logger.warning(f"[Obj2] Failed to load importance model: {exc}")
        return False


def get_importance_score(
    layer: str,
    age_days: int = 0,
    access_count: int = 0,
    text_length: int = 100,
    sentiment_score: float = 0.0,
) -> float:
    """Predict v_E importance score for a new memory.

    If no model is loaded, falls back to the paper's random initialization:
        v_E ∈ {40, 60, 80} with layer-dependent probabilities.

    Args:
        layer:           Memory layer name (short/mid/long/reflection).
        age_days:        Days since memory creation.
        access_count:    Times memory was retrieved.
        text_length:     Character count of memory text.
        sentiment_score: Sentiment score in [-1, 1].

    Returns:
        Float in [40, 80] — the initial importance value v_E.
    """
    global _bundle

    # Try to auto-load model on first call
    if _bundle is None:
        learned = os.getenv("LEARNED_IMPORTANCE", "false").lower() == "true"
        if learned:
            load_model()

    # If no model → fallback to paper random
    if _bundle is None:
        return _random_fallback(layer)

    try:
        from .trainer import build_feature_vector

        features = build_feature_vector(
            layer=layer,
            age_days=age_days,
            access_count=access_count,
            text_length=text_length,
            sentiment_score=sentiment_score,
        )

        X = np.array([features], dtype=np.float32)
        X_scaled = _bundle["scaler"].transform(X)
        prob = _bundle["clf"].predict_proba(X_scaled)[0][1]  # P(label=1) = profitable

        # Map probability → v_E in [40, 80]
        v_E = 40.0 + (prob * 40.0)

        logger.debug(f"[Obj2] Predicted v_E={v_E:.1f} (prob={prob:.3f}) "
                      f"for layer={layer}, age={age_days}")
        return float(v_E)

    except Exception as exc:
        logger.warning(f"[Obj2] Inference failed, falling back to random: {exc}")
        return _random_fallback(layer)


def _random_fallback(layer: str) -> float:
    """Paper's original random importance initialization.

    Returns value in {40, 60, 80} based on layer probabilities.
    On a [0, 1] scale, these correspond to {0.4, 0.6, 0.8}.
    Here we return them on a [40, 80] scale for direct v_E usage.
    """
    layer_probs = {
        "short":      [0.6, 0.3, 0.1],
        "shallow":    [0.6, 0.3, 0.1],
        "mid":        [0.2, 0.6, 0.2],
        "intermediate": [0.2, 0.6, 0.2],
        "long":       [0.1, 0.3, 0.6],
        "deep":       [0.1, 0.3, 0.6],
        "reflection": [0.1, 0.3, 0.6],
    }
    probs = layer_probs.get(layer.lower(), [0.33, 0.34, 0.33])
    v_E = float(np.random.choice([40, 60, 80], p=probs))
    return v_E


def is_model_loaded() -> bool:
    """Check if the importance model is currently loaded."""
    return _bundle is not None


def get_model_info() -> dict:
    """Return info about the loaded model (for UI/diagnostics)."""
    if _bundle is None:
        return {"loaded": False, "fallback": "random (paper default)"}
    return {
        "loaded": True,
        "model_type": _bundle.get("model_type", "unknown"),
        "accuracy": _bundle.get("accuracy", 0),
        "n_samples": _bundle.get("n_samples", 0),
    }
