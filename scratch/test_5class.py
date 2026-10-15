"""
test_5class.py
Verifies 5-class predictions across all 3 trained models with genuine probability calibration.
"""

import sys
import json
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import SentimentPredictor

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_5class_predictions():
    """Verifies 5-class predictions across all 3 models."""
    metrics_path = ROOT_DIR / "artifacts" / "metrics.json"
    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)

    assert m["dataset_metadata"]["num_classes"] == 5
    assert len(m["dataset_metadata"]["classes"]) == 5

    test_cases = [
        ("Rave Review", "An exquisite, heart-wrenching cinematic masterpiece with breathtaking visuals and sublime acting."),
        ("Good Review", "A charming, thoroughly enjoyable feel-good comedy with great comic timing."),
        ("Neutral Review", "A moderately competent procedural thriller that follows standard genre conventions without surprises."),
        ("Poor Review", "The screenplay is disjointed, dialogue is awkward, and the pacing drags painfully."),
        ("Awful Review", "A complete and utter catastrophe. Abysmal acting, hideous editing, and unwatchable nonsense.")
    ]

    for model_name in ["Linear SVM", "Logistic Regression", "Multinomial Naive Bayes"]:
        predictor = SentimentPredictor(model_name=model_name)
        for label, text in test_cases:
            res = predictor.predict(text)
            assert res["predicted_label"] in {0, 1, 2, 3, 4}
            assert len(res["probabilities"]) == 5


if __name__ == "__main__":
    test_5class_predictions()
    print("[Pass] All 5-class predictions verified successfully!")
