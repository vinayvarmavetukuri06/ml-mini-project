"""
conftest.py
Shared pytest fixtures for the Movie Review Sentiment Analysis test suite.
Reuses existing saved models and avoids retraining during test runs.
"""

import sys
import json
from pathlib import Path
import pytest

# Ensure project root is at the front of sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import SentimentPredictor


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Returns the workspace project root path."""
    return ROOT_DIR


@pytest.fixture(scope="session")
def metrics_data(project_root) -> dict:
    """Loads artifacts/metrics.json for inspection."""
    metrics_file = project_root / "artifacts" / "metrics.json"
    assert metrics_file.exists(), f"Metrics file missing at {metrics_file}"
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def default_predictor():
    """Returns a cached SentimentPredictor instance loaded with the champion model."""
    return SentimentPredictor()


@pytest.fixture(scope="session")
def all_predictors():
    """Returns a dictionary of SentimentPredictors for each supported classifier."""
    predictors = {}
    for name in SentimentPredictor.AVAILABLE_MODELS:
        predictors[name] = SentimentPredictor(model_name=name)
    return predictors
