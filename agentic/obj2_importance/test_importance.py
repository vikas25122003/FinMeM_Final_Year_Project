"""
Objective 2 — Step 5: End-to-End Test

Creates fake reflection records, trains a classifier on them,
and verifies inference produces valid importance scores.

Run:
    python -m agentic.obj2_importance.test_importance
"""

import os
import sys
import shutil
import tempfile
import numpy as np


def test_objective_2():
    """End-to-end test for Objective 2: Learned Importance Scoring."""

    print("=" * 60)
    print("  🧪 Objective 2 — Learned Importance Scoring Test")
    print("=" * 60)

    # Use temp dirs to avoid polluting real data
    test_log_dir = os.path.join(tempfile.gettempdir(), "finmem_test_reflections")
    test_model_path = os.path.join(tempfile.gettempdir(), "finmem_test_importance.pkl")

    # Clean up from previous test runs
    if os.path.exists(test_log_dir):
        shutil.rmtree(test_log_dir)

    try:
        # ── Step 1: Create fake reflection records ──
        print("\n  Step 1: Creating fake reflection records...")

        from agentic.obj2_importance.logger import log_reflection, load_all_reflections

        fake_records = [
            # Profitable decisions (next-day will be positive)
            {"date": "2022-03-14", "ticker": "TSLA", "decision": "BUY",
             "memory_ids_used": [1, 2, 3], "rationale": "Strong rally expected based on earnings beat"},
            {"date": "2022-03-15", "ticker": "TSLA", "decision": "BUY",
             "memory_ids_used": [4, 5], "rationale": "Positive momentum and institutional buying surge"},
            {"date": "2022-03-16", "ticker": "TSLA", "decision": "HOLD",
             "memory_ids_used": [6], "rationale": "Market uncertain but no reason to sell"},
            {"date": "2022-03-17", "ticker": "TSLA", "decision": "BUY",
             "memory_ids_used": [7, 8], "rationale": "Strong bull signal from technical patterns"},
            {"date": "2022-03-18", "ticker": "TSLA", "decision": "SELL",
             "memory_ids_used": [9, 10], "rationale": "Taking profits after crash warning"},
            {"date": "2022-03-21", "ticker": "TSLA", "decision": "BUY",
             "memory_ids_used": [11, 12], "rationale": "Dip buy opportunity after correction"},
            {"date": "2022-03-22", "ticker": "TSLA", "decision": "BUY",
             "memory_ids_used": [13, 14, 15], "rationale": "Growth outlook remains positive"},
            {"date": "2022-03-23", "ticker": "TSLA", "decision": "SELL",
             "memory_ids_used": [16], "rationale": "Risk off signal from bond market decline"},
            {"date": "2022-03-24", "ticker": "TSLA", "decision": "BUY",
             "memory_ids_used": [17, 18], "rationale": "Recovery play on oversold conditions"},
            {"date": "2022-03-25", "ticker": "TSLA", "decision": "HOLD",
             "memory_ids_used": [19, 20], "rationale": "Weekend hold, no clear signal"},
        ]

        for r in fake_records:
            log_reflection(log_dir=test_log_dir, **r)

        reflections = load_all_reflections(log_dir=test_log_dir)
        assert len(reflections) == 10, f"Expected 10, got {len(reflections)}"
        print(f"    ✓ Created {len(reflections)} reflection records")

        # ── Step 2: Build feature vectors and synthetic labels ──
        print("\n  Step 2: Building feature vectors with synthetic labels...")

        from agentic.obj2_importance.trainer import build_feature_vector, train_classifier

        # Build synthetic training data (instead of real yfinance calls)
        X_list = []
        y_list = []

        # Create diverse training data
        np.random.seed(42)
        for i in range(50):
            layer = np.random.choice(["short", "mid", "long"])
            age = np.random.randint(0, 30)
            access = np.random.randint(0, 10)
            text_len = np.random.randint(50, 500)
            sentiment = np.random.uniform(-1, 1)

            features = build_feature_vector(layer, age, access, text_len, sentiment)
            X_list.append(features)

            # Synthetic label: higher sentiment and shorter layer → more likely profitable
            prob = 0.5 + 0.2 * sentiment - 0.1 * (1 if layer == "long" else 0) + 0.05 * (access / 10)
            label = 1 if np.random.random() < prob else 0
            y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        assert X.shape == (50, 7), f"Expected shape (50, 7), got {X.shape}"
        print(f"    ✓ Built {len(X)} feature vectors (shape: {X.shape})")

        # ── Step 3: Train classifier ──
        print("\n  Step 3: Training classifier...")

        bundle = train_classifier(X, y, model_save_path=test_model_path)
        assert bundle, "Training returned empty bundle"
        assert "clf" in bundle, "Bundle missing 'clf'"
        assert "scaler" in bundle, "Bundle missing 'scaler'"
        assert os.path.exists(test_model_path), f"Model not saved at {test_model_path}"
        print(f"    ✓ Classifier trained (accuracy: {bundle.get('accuracy', 0):.3f})")

        # ── Step 4: Test inference ──
        print("\n  Step 4: Testing inference...")

        from agentic.obj2_importance.inference import load_model, get_importance_score, is_model_loaded

        loaded = load_model(test_model_path)
        assert loaded, "Failed to load model"
        assert is_model_loaded(), "Model not marked as loaded"

        # Test multiple inputs
        test_cases = [
            ("short", 5, 2, 250, 0.3),
            ("mid", 10, 5, 400, -0.5),
            ("long", 30, 0, 100, 0.0),
            ("reflection", 1, 8, 300, 0.8),
        ]

        for layer, age, access, tlen, sent in test_cases:
            score = get_importance_score(layer, age, access, tlen, sent)
            assert isinstance(score, float), f"Score should be float, got {type(score)}"
            assert 40.0 <= score <= 80.0, f"Score {score} out of range [40, 80]"
            print(f"    get_importance_score('{layer}', {age}, {access}, {tlen}, {sent}) → {score:.1f}")

        print(f"\n    ✓ All inference scores in valid range [40, 80]")

        # ── Step 5: Verify feature vector shape ──
        print("\n  Step 5: Verifying feature engineering...")

        fv = build_feature_vector("short", 10, 3, 200, 0.5)
        assert len(fv) == 7, f"Expected 7 features, got {len(fv)}"
        assert fv[:3] == [1.0, 0.0, 0.0], f"Wrong layer encoding for 'short'"

        fv2 = build_feature_vector("mid", 0, 0, 0, 0)
        assert fv2[:3] == [0.0, 1.0, 0.0], f"Wrong layer encoding for 'mid'"

        fv3 = build_feature_vector("long", 0, 0, 0, 0)
        assert fv3[:3] == [0.0, 0.0, 1.0], f"Wrong layer encoding for 'long'"

        print(f"    ✓ Feature vectors correct shape and encoding")

        print("\n" + "=" * 60)
        print("  ✅ Objective 2 test passed.")
        print("=" * 60)

    finally:
        # Cleanup
        if os.path.exists(test_log_dir):
            shutil.rmtree(test_log_dir)
        if os.path.exists(test_model_path):
            os.remove(test_model_path)


if __name__ == "__main__":
    # Ensure we can import from project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    test_objective_2()
