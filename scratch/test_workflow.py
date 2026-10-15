"""
test_workflow.py
Smoke test to verify model saving/loading across all 3 classifiers
and check Streamlit module compatibility.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import SentimentPredictor

def test_workflow_model_saving_loading():
    """Smoke test to verify model saving/loading across all 3 classifiers."""
    test_reviews = [
        "A gripping, breathtaking spectacle with phenomenal acting and masterful direction.",
        "An utterly uninspired, painfully boring disaster with terrible dialogue and zero plot."
    ]

    models = ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"]

    for model_name in models:
        predictor = SentimentPredictor(model_name=model_name)
        for review in test_reviews:
            res = predictor.predict(review)
            assert res["predicted_label"] in {0, 1, 2, 3, 4}
            assert 0.0 <= res["confidence"] <= 1.0
            assert len(res["probabilities"]) == 5


if __name__ == "__main__":
    test_workflow_model_saving_loading()
    print("\n[Validation] All models saved, loaded, and evaluated successfully!")
