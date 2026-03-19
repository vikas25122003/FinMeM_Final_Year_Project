"""
Objective 1 — Step 2: Market Regime Classifiers

Two implementations exposing the same `.predict(features) -> str` interface:

  ThresholdRegimeClassifier (baseline, interpretable)
    Rule-based. Classify by annualized vol + 20-day momentum.
    Thresholds grounded in quant finance (VIX 30 ≈ 35% annualized vol).
    No training required. Deterministic.

  HMMRegimeClassifier (our research contribution)
    Gaussian HMM with 3 hidden states fitted by Baum-Welch EM on
    historical daily log-returns. States auto-labelled after fitting:
      lowest mean return  → CRISIS
      middle mean return  → SIDEWAYS
      highest mean return → BULL
    LEARNS regime boundaries from data rather than assuming them.
    Requires one-time training via train_hmm.py (< 30 sec locally).

Both classifiers output: "BULL" | "SIDEWAYS" | "CRISIS"

Usage:
    from agentic.obj1_regime.classifier import get_classifier
    clf    = get_classifier("hmm")          # or "threshold"
    regime = clf.predict(features)          # features from compute_features()
"""

import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

HMM_MODEL_PATH = os.environ.get("HMM_MODEL_PATH", "./models/hmm_regime.pkl")
N_COMPONENTS   = 3     # BULL, SIDEWAYS, CRISIS
N_ITER         = 200   # Baum-Welch EM iterations


# ── Classifier 1: Threshold (Rule-Based Baseline) ────────────────────────────

class ThresholdRegimeClassifier:
    """
    Threshold-based market regime classifier.

    Panel defence rationale for thresholds:
      ≥35% annualized vol  → CRISIS   (≈ VIX 30, widely-used fear gauge cutoff)
      ≤20% annualized vol  AND
      ≥5% 20d return       → BULL     (calm + trending = sustained bull)
      Else                 → SIDEWAYS (paper defaults apply)
    """

    CRISIS_VOL_THRESHOLD    = 0.35
    BULL_VOL_THRESHOLD      = 0.20
    BULL_MOMENTUM_THRESHOLD = 0.05

    def predict(self, features: Dict) -> str:
        vol_20d      = features.get("vol_20d", 0.20)
        momentum_20d = features.get("momentum_20d", 0.0)

        if vol_20d >= self.CRISIS_VOL_THRESHOLD:
            return "CRISIS"
        elif vol_20d <= self.BULL_VOL_THRESHOLD and momentum_20d >= self.BULL_MOMENTUM_THRESHOLD:
            return "BULL"
        else:
            return "SIDEWAYS"

    def predict_proba(self, features: Dict) -> Dict[str, float]:
        """Hard probabilities (1.0 for chosen regime, 0.0 for others)."""
        regime = self.predict(features)
        return {r: (1.0 if r == regime else 0.0) for r in ["BULL", "SIDEWAYS", "CRISIS"]}


# ── Classifier 2: HMM (Research Contribution) ────────────────────────────────

class HMMRegimeClassifier:
    """
    Gaussian Hidden Markov Model for unsupervised market regime detection.

    Why HMM over threshold rules?
    - HMM LEARNS regime boundaries from data (no hardcoded vol threshold)
    - Captures regime persistence via the transition matrix A (states are sticky)
    - Handles noisy returns via Gaussian emission: r_t ~ N(μ_k, σ_k²)
    - Naturally discovers 3 latent states without supervision

    Architecture:
        Hidden states: S ∈ {0, 1, 2}  (latent regimes, auto-labelled post-fit)
        Observations:  daily log-returns r_t (univariate, 60-day window)
        Parameters:    π (initial probs), A (transition matrix), μ, Σ (Gaussians)
        Training:      Baum-Welch EM algorithm (forward-backward)
        Inference:     Viterbi decoding → predict last state in sequence

    State labelling (post-training):
        Sort states by mean return:
          lowest  mean → CRISIS   (negative daily drift)
          middle  mean → SIDEWAYS (near-zero drift)
          highest mean → BULL     (positive daily drift)

    Key advantage over threshold: The HMM can detect a calm but trending
    market (BULL) even if vol is 22% — the threshold rule would miss this.
    The HMM see the pattern in the full 60-day sequence, not just today's vol.
    """

    def __init__(self):
        self.model        = None
        self.state_labels: Dict[int, str] = {}
        self.is_fitted    = False

    def fit(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        n_components: int = N_COMPONENTS,
        n_iter: int = N_ITER,
    ) -> "HMMRegimeClassifier":
        """
        Fit Gaussian HMM on historical daily log-returns.

        Args:
            ticker:       Stock symbol (e.g. "TSLA")
            start_date:   Training start  "YYYY-MM-DD"
            end_date:     Training end    "YYYY-MM-DD"
                          IMPORTANT: use data BEFORE your FinMEM train period
                          to avoid lookahead bias.
            n_components: Number of hidden states (3 = BULL/SIDEWAYS/CRISIS)
            n_iter:       Baum-Welch EM iterations

        Returns:
            self (for chaining)
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("Run: pip install hmmlearn")

        import pandas as pd
        import yfinance as yf

        logger.info(f"[HMM] Fetching data: {ticker}  {start_date} → {end_date}")
        df = yf.download(ticker, start=start_date, end=end_date,
                         auto_adjust=True, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 60:
            raise ValueError(f"Need ≥60 trading days, got {len(df)}")

        import pandas as pd
        close       = df["Close"].squeeze()
        log_returns = np.log(close / close.shift(1)).dropna().values
        log_returns = log_returns.reshape(-1, 1)           # (T, 1) for hmmlearn

        logger.info(f"[HMM] Fitting GaussianHMM with {n_components} states on {len(log_returns)} observations...")

        model = GaussianHMM(
            n_components    = n_components,
            covariance_type = "full",
            n_iter          = n_iter,
            random_state    = 42,
            tol             = 1e-4,
        )
        model.fit(log_returns)

        # Auto-label states by mean return (ascending order)
        means         = model.means_.flatten()
        sorted_states = np.argsort(means)   # lowest → highest

        self.state_labels = {
            int(sorted_states[0]): "CRISIS",    # most negative mean return
            int(sorted_states[1]): "SIDEWAYS",  # middle
            int(sorted_states[2]): "BULL",      # most positive mean return
        }
        self.model     = model
        self.is_fitted = True

        logger.info(f"[HMM] Fitted. State labels: {self.state_labels}")
        for i in range(n_components):
            label = self.state_labels[i]
            mean  = float(model.means_[i, 0])
            std   = float(np.sqrt(model.covars_[i, 0, 0]))
            logger.info(f"[HMM]   State {i} ({label}): μ={mean:.5f}  σ={std:.5f}")

        return self

    def save(self, path: str = HMM_MODEL_PATH) -> None:
        """Save fitted HMM + state labels to disk as pickle."""
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "labels": self.state_labels}, path)
        logger.info(f"[HMM] Saved to {path}")

    def load(self, path: str = HMM_MODEL_PATH) -> "HMMRegimeClassifier":
        """Load a previously fitted HMM from disk."""
        import joblib
        if not Path(path).exists():
            raise FileNotFoundError(
                f"HMM model not found at '{path}'.\n"
                "Train it first:  python3 -m agentic.obj1_regime.train_hmm"
            )
        bundle            = joblib.load(path)
        self.model        = bundle["model"]
        self.state_labels = bundle["labels"]
        self.is_fitted    = True
        logger.info(f"[HMM] Loaded from {path}")
        return self

    def predict(self, features: Dict) -> str:
        """
        Predict regime from features dict.

        Uses the 60-element `returns_seq` from features (log-returns sequence).
        Falls back to ThresholdRegimeClassifier if HMM not fitted.

        Args:
            features: Dict from compute_features() — must have `returns_seq`.

        Returns:
            Regime string: "BULL" | "SIDEWAYS" | "CRISIS"
        """
        if not self.is_fitted:
            logger.warning("[HMM] Not fitted — falling back to threshold classifier")
            return ThresholdRegimeClassifier().predict(features)

        returns_seq = np.array(
            features.get("returns_seq", [0.0] * 60),
            dtype=float,
        ).reshape(-1, 1)

        # Replace inf/nan (safety)
        returns_seq = np.where(np.isfinite(returns_seq), returns_seq, 0.0)

        try:
            states     = self.model.predict(returns_seq)
            last_state = int(states[-1])
            regime     = self.state_labels.get(last_state, "SIDEWAYS")
            logger.debug(f"[HMM] Predicted state={last_state} → {regime}")
            return regime

        except Exception as exc:
            logger.warning(f"[HMM] predict() failed: {exc} — falling back to threshold")
            return ThresholdRegimeClassifier().predict(features)

    def predict_proba(self, features: Dict) -> Dict[str, float]:
        """Posterior probability of each regime from the last time-step."""
        if not self.is_fitted:
            return ThresholdRegimeClassifier().predict_proba(features)

        returns_seq = np.array(
            features.get("returns_seq", [0.0] * 60), dtype=float
        ).reshape(-1, 1)
        returns_seq = np.where(np.isfinite(returns_seq), returns_seq, 0.0)

        try:
            posteriors = self.model.predict_proba(returns_seq)
            last_probs = posteriors[-1]
            return {
                self.state_labels.get(i, "SIDEWAYS"): float(last_probs[i])
                for i in range(N_COMPONENTS)
            }
        except Exception:
            return {"BULL": 0.33, "SIDEWAYS": 0.34, "CRISIS": 0.33}


# ── Factory ──────────────────────────────────────────────────────────────────

def get_classifier(mode: str = "threshold"):
    """
    Factory — return the appropriate regime classifier.

    Args:
        mode: "threshold" (default, no training needed)
              "hmm"       (research contribution, run train_hmm.py first)

    Returns:
        Fitted classifier with .predict(features) -> str
    """
    if mode == "threshold":
        return ThresholdRegimeClassifier()

    elif mode == "hmm":
        clf = HMMRegimeClassifier()
        try:
            clf.load(HMM_MODEL_PATH)
        except FileNotFoundError as e:
            logger.warning(str(e))
            logger.warning("[HMM] Returning an unfitted classifier — will fall back to threshold.")
        return clf

    else:
        raise ValueError(f"Unknown classifier mode: '{mode}'. Use 'threshold' or 'hmm'.")
