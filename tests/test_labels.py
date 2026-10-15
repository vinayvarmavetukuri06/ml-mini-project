"""
test_labels.py
Tests the five-class sentiment taxonomy, integer-to-string mapping,
persisted label artifacts, and dataset location resolution.
"""

from pathlib import Path
import joblib
from src.data_loader import LABEL_MAP_5_CLASS, find_dataset_file


def test_label_map_constant():
    """Verifies LABEL_MAP_5_CLASS constant defines the exact 5 Rotten Tomatoes sentiment classes."""
    assert len(LABEL_MAP_5_CLASS) == 5
    assert LABEL_MAP_5_CLASS[0] == "Very Negative"
    assert LABEL_MAP_5_CLASS[1] == "Negative"
    assert LABEL_MAP_5_CLASS[2] == "Neutral"
    assert LABEL_MAP_5_CLASS[3] == "Positive"
    assert LABEL_MAP_5_CLASS[4] == "Very Positive"


def test_persisted_label_map_artifact(project_root):
    """Verifies artifacts/models/label_map.joblib matches the 5-class mapping."""
    label_map_path = project_root / "artifacts" / "models" / "label_map.joblib"
    assert label_map_path.exists(), f"label_map.joblib missing at {label_map_path}"

    loaded_map = joblib.load(label_map_path)
    assert isinstance(loaded_map, dict)
    assert len(loaded_map) == 5
    for k, v in LABEL_MAP_5_CLASS.items():
        assert loaded_map[k] == v


def test_dataset_discovery(project_root):
    """Verifies find_dataset_file resolves the training dataset in data/."""
    ds_path = find_dataset_file()
    assert ds_path is not None, "Dataset file was not located by find_dataset_file()"
    assert ds_path.is_file()
    assert ds_path.name in ["train.tsv", "movie_reviews.csv"]
    assert ds_path.stat().st_size > 0
