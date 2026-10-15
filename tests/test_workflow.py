"""
test_workflow.py
Smoke test verifying model loading, inference pipeline, keyword attribution,
and batch predictions across all 3 trained classifiers.
"""

import pytest
from src.predict import SentimentPredictor


def test_all_models_loadable_and_functional(all_predictors):
    """Verifies all 3 models load successfully and perform inference."""
    test_reviews = [
        "A gripping, breathtaking spectacle with phenomenal acting and masterful direction.",
        "An utterly uninspired, painfully boring disaster with terrible dialogue and zero plot."
    ]

    for model_name, predictor in all_predictors.items():
        assert predictor.active_model_name == model_name
        for review in test_reviews:
            res = predictor.predict(review)
            assert res["predicted_label"] in {0, 1, 2, 3, 4}
            assert isinstance(res["sentiment"], str)
            assert len(res["sentiment"]) > 0
            assert 0.0 <= res["confidence"] <= 1.0


def test_keyword_attributions(default_predictor):
    """Verifies that TF-IDF keyword weights are extracted for review tokens."""
    sample = "Masterpiece cinematic spectacle with wonderful visuals."
    res = default_predictor.predict(sample)

    assert "top_keywords" in res
    keywords = res["top_keywords"]
    assert isinstance(keywords, list)
    assert len(keywords) > 0

    for item in keywords:
        assert "word" in item
        assert "tfidf_weight" in item
        assert item["tfidf_weight"] > 0


def test_batch_prediction_workflow(default_predictor):
    """Verifies predict_batch processes a list of inputs identically to predict."""
    reviews = [
        "Brilliant performance and directing.",
        "Dull and completely uninteresting."
    ]
    batch_results = default_predictor.predict_batch(reviews)

    assert len(batch_results) == 2
    for i, res in enumerate(batch_results):
        single_res = default_predictor.predict(reviews[i])
        assert res["predicted_label"] == single_res["predicted_label"]
        assert res["sentiment"] == single_res["sentiment"]
        assert res["confidence"] == single_res["confidence"]
