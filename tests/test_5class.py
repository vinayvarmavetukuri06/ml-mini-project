"""
test_5class.py
Verifies 5-class predictions across all 3 trained models with genuine probability calibration.
Addresses requirement for 5-class sentiment taxonomy and label mapping tests.
"""

import pytest
from src.predict import SentimentPredictor


def test_five_class_metadata_integrity(metrics_data):
    """Verifies dataset metadata and model artifacts adhere to the 5-class taxonomy."""
    meta = metrics_data.get("dataset_metadata", {})
    assert meta.get("num_classes") == 5, f"Expected 5 classes, got {meta.get('num_classes')}"

    expected_classes = [
        "Very Negative",
        "Negative",
        "Neutral",
        "Positive",
        "Very Positive"
    ]
    assert meta.get("classes") == expected_classes, "Classes list does not match 5-class standard"

    dist = meta.get("class_distribution", {})
    assert len(dist) == 5, "Class distribution must contain 5 classes"
    for cls_name in expected_classes:
        assert dist.get(cls_name, 0) > 0, f"Class {cls_name} has zero or missing count"


@pytest.mark.parametrize("model_name", [
    "Linear SVM",
    "Logistic Regression",
    "Multinomial Naive Bayes"
])
def test_five_class_prediction_structure(model_name, all_predictors):
    """Verifies that each model outputs valid 5-class labels and calibrated probability distributions."""
    predictor = all_predictors[model_name]
    review = "A charming, thoroughly enjoyable feel-good comedy with great comic timing."
    res = predictor.predict(review)

    assert "predicted_label" in res
    assert res["predicted_label"] in {0, 1, 2, 3, 4}
    assert res["sentiment"] in {
        "Very Negative", "Negative", "Neutral", "Positive", "Very Positive"
    }
    assert 0.0 <= res["confidence"] <= 1.0

    probs = res.get("probabilities", {})
    assert len(probs) == 5, f"Model {model_name} did not return 5 class probabilities"
    for expected_name in ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]:
        assert expected_name in probs
        assert 0.0 <= probs[expected_name] <= 1.0

    prob_sum = sum(probs.values())
    assert pytest.approx(prob_sum, abs=0.02) == 1.0, f"Probabilities for {model_name} sum to {prob_sum}, expected ~1.0"


def test_sentiment_distinct_extremes(default_predictor):
    """Verifies model correctly differentiates strongly polarized reviews."""
    rave = "An exquisite, heart-wrenching cinematic masterpiece with breathtaking visuals and sublime acting."
    awful = "A complete and utter catastrophe. Abysmal acting, hideous editing, and unwatchable nonsense."

    res_rave = default_predictor.predict(rave)
    res_awful = default_predictor.predict(awful)

    # Rave review should be Positive (3) or Very Positive (4)
    assert res_rave["predicted_label"] in {3, 4}, f"Expected positive label, got {res_rave}"

    # Awful review should be Negative (1) or Very Negative (0)
    assert res_awful["predicted_label"] in {0, 1}, f"Expected negative label, got {res_awful}"
