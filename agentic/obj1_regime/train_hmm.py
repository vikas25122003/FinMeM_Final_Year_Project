"""
train_hmm.py — One-time script to fit and save the HMM Regime Classifier.

Run this ONCE before using ADAPTIVE_Q_MODE=hmm.

The HMM is trained on TSLA daily log-returns from 2019-01-01 → 2022-03-13.
This is strictly before the FinMEM training period (2022-03-14 → 2022-06-15)
→ no lookahead bias from the training data.

Training takes < 30 seconds locally. No GPU needed.

Usage:
    python3 -m agentic.obj1_regime.train_hmm
    # or:
    python3 agentic/obj1_regime/train_hmm.py

Output:
    ./models/hmm_regime.pkl     ← HMM model + state labels
"""

import os
import sys
import logging

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def train(
    ticker: str = "TSLA",
    start_date: str = "2019-01-01",
    end_date: str   = "2022-03-13",   # strictly before FinMEM train start
    n_states: int   = 3,
    n_iter: int     = 200,
    save_path: str  = "./models/hmm_regime.pkl",
) -> None:
    """
    Fit and save a GaussianHMM regime classifier.

    Args:
        ticker:     Stock to train on. Use TSLA for FinMEM comparisons.
        start_date: Training data start (YYYY-MM-DD).
        end_date:   Training data end. Must be BEFORE FinMEM train period.
        n_states:   Number of HMM hidden states (3 = BULL/SIDEWAYS/CRISIS).
        n_iter:     Baum-Welch EM max iterations.
        save_path:  Where to save the fitted model pickle.
    """
    from agentic.obj1_regime.classifier import HMMRegimeClassifier

    print("\n" + "="*55)
    print("  Training HMM Regime Classifier (Objective 1)")
    print("="*55)
    print(f"\n  Ticker     : {ticker}")
    print(f"  Train Data : {start_date} → {end_date}")
    print(f"  HMM States : {n_states}  (BULL / SIDEWAYS / CRISIS)")
    print(f"  EM Iters   : {n_iter}")
    print(f"  Save Path  : {save_path}\n")

    clf = HMMRegimeClassifier()
    clf.fit(
        ticker     = ticker,
        start_date = start_date,
        end_date   = end_date,
        n_components = n_states,
        n_iter       = n_iter,
    )
    clf.save(save_path)

    # Pretty-print results
    print("\n  ─── State Summary ─────────────────────────────────")
    import numpy as np
    for state_id, label in sorted(clf.state_labels.items()):
        mean = float(clf.model.means_[state_id, 0])
        std  = float(np.sqrt(clf.model.covars_[state_id, 0, 0]))
        ann_vol = std * (252 ** 0.5) * 100
        print(
            f"  State {state_id} ({label:<9}): "
            f"μ_daily={mean*100:+.4f}%   "
            f"σ_daily={std*100:.4f}%   "
            f"Ann.Vol≈{ann_vol:.1f}%"
        )

    print("\n  ─── Transition Matrix (rows = from, cols = to) ────")
    labels_ordered = ["CRISIS", "SIDEWAYS", "BULL"]
    state_order    = [k for label in labels_ordered for k, v in clf.state_labels.items() if v == label]
    A              = clf.model.transmat_
    print(f"  {'':>12}", end="")
    for lbl in labels_ordered:
        print(f"  {lbl:>9}", end="")
    print()
    for i, from_label in enumerate(labels_ordered):
        state_i = state_order[i]
        print(f"  {from_label:>12}", end="")
        for j, to_label in enumerate(labels_ordered):
            state_j = state_order[j]
            print(f"  {A[state_i, state_j]:>9.4f}", end="")
        print()

    print("\n  The transition matrix shows regime PERSISTENCE.")
    print("  High diagonal values mean the market stays in one regime for many days.")

    print(f"\n{'='*55}")
    print(f"\n  ✅ HMM saved → {save_path}")
    print(f"\n  Now run with:")
    print(f"    ADAPTIVE_Q_MODE=hmm python3 run_obj1.py --ticker TSLA ...")
    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HMM Regime Classifier")
    parser.add_argument("--ticker",     default="TSLA",       help="Stock ticker for training data")
    parser.add_argument("--start-date", default="2019-01-01", help="Training start date")
    parser.add_argument("--end-date",   default="2022-03-13", help="Training end date (before FinMEM train)")
    parser.add_argument("--n-states",   default=3,  type=int, help="Number of HMM hidden states")
    parser.add_argument("--n-iter",     default=200,type=int, help="Baum-Welch EM iterations")
    parser.add_argument("--save-path",  default="./models/hmm_regime.pkl", help="Output path")
    args = parser.parse_args()

    train(
        ticker     = args.ticker,
        start_date = args.start_date,
        end_date   = args.end_date,
        n_states   = args.n_states,
        n_iter     = args.n_iter,
        save_path  = args.save_path,
    )
