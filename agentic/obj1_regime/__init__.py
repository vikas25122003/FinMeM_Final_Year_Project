"""
Objective 1: Regime-Conditioned Adaptive Memory Decay

Replaces FinMEM's fixed Q constants in the recency formula:
    S_recency = e^(-delta / Q_l)

with regime-conditioned Q-values derived from daily market features.

Usage:
    from agentic.obj1_regime.features import compute_features
    from agentic.obj1_regime.classifier import get_classifier
    from agentic.obj1_regime.q_table import get_Q

    features = compute_features("TSLA", "2022-10-25")
    regime   = get_classifier().predict(features)   # "CRISIS"
    Q        = get_Q(regime, "shallow")              # 5
"""
