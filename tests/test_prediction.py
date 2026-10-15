"""
test_prediction.py
Tests SentimentPredictor model loading, model switching, prediction output schema,
and handling of empty, whitespace, and out-of-vocabulary inputs.
"""

import pytest
from src.predict import SentimentPredictor


def test_default_model_loading_selects_champion():
    """Verifies initializing SentimentPredictor without arguments automatically selects the champion model."""
    predictor = SentimentPredictor()
    assert predictor.active_model_name == "Linear SVM"
    assert predictor.model is not None
    assert predictor.vectorizer is not None


def test_explicit_model_switching():
    """Verifies switching models updates active_model_name and active classifier instance."""
    predictor = SentimentPredictor(model_name="Logistic Regression")
    assert predictor.active_model_name == "Logistic Regression"

    predictor.switch_model("Multinomial Naive Bayes")
    assert predictor.active_model_name == "Multinomial Naive Bayes"

    predictor.switch_model("Linear SVM")
    assert predictor.active_model_name == "Linear SVM"


def test_unknown_model_loading_fallback():
    """Verifies requesting an unknown/unsupported model name gracefully falls back to an available model."""
    predictor = SentimentPredictor(model_name="NonExistentModel999")
    assert predictor.active_model_name in SentimentPredictor.AVAILABLE_MODELS
    assert predictor.model is not None


def test_prediction_output_schema(default_predictor):
    """Verifies the prediction dictionary contains all documented fields and correct types."""
    review = "A compelling and emotionally moving story with spectacular performances."
    res = default_predictor.predict(review)

    assert res["text"] == review
    assert res["model_used"] == "Linear SVM"
    assert isinstance(res["predicted_label"], int)
    assert res["predicted_label"] in {0, 1, 2, 3, 4}
    assert res["sentiment"] in {"Very Negative", "Negative", "Neutral", "Positive", "Very Positive"}
    assert isinstance(res["confidence"], float)
    assert 0.0 <= res["confidence"] <= 1.0

    probs = res["probabilities"]
    assert isinstance(probs, dict)
    assert len(probs) == 5

    keywords = res["top_keywords"]
    assert isinstance(keywords, list)
    assert len(keywords) > 0


@pytest.mark.parametrize("empty_input", ["", "   ", "\t\n  ", None])
def test_empty_or_whitespace_review_handling(empty_input, default_predictor):
    """Verifies empty or whitespace-only inputs are handled gracefully without crashing."""
    res = default_predictor.predict(empty_input)
    assert res["predicted_label"] is None
    assert res["sentiment"] == "Unknown"
    assert res["confidence"] == 0.0
    assert res["probabilities"] == {}
    assert res["top_keywords"] == []


def test_out_of_vocabulary_review_handling(default_predictor):
    """Verifies gibberish or entirely out-of-vocabulary text does not crash the predictor."""
    oov_text = "qwertyxyz123987 abcdefghij9999 zzzxxxcccvvv"
    res = default_predictor.predict(oov_text)
    assert res["predicted_label"] in {0, 1, 2, 3, 4}
    assert res["sentiment"] in {"Very Negative", "Negative", "Neutral", "Positive", "Very Positive"}
    assert 0.0 <= res["confidence"] <= 1.0
