"""
predict.py
Inference module for movie review sentiment classification and model interpretability.
Loads trained models, vectorizer, and maps predictions to probabilities and keyword attributions.
"""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path before unpickling artifacts
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Any, List, Optional
import joblib
import numpy as np


class SentimentPredictor:
    """
    Production-ready inference wrapper with multi-model support
    and keyword sentiment explainability.
    """

    AVAILABLE_MODELS = [
        "Linear SVM",
        "Logistic Regression",
        "Multinomial Naive Bayes"
    ]

    def __init__(self, model_name: Optional[str] = None):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.models_dir = self.root_dir / "artifacts" / "models"

        # Load TF-IDF vectorizer
        vec_path = self.models_dir / "tfidf_vectorizer.joblib"
        if not vec_path.exists():
            raise FileNotFoundError(
                f"Vectorizer not found at {vec_path}. Please execute 'src/train.py' first."
            )
        self.vectorizer = joblib.load(vec_path)

        # Load label mapping
        map_path = self.models_dir / "label_map.joblib"
        if map_path.exists():
            self.label_map = joblib.load(map_path)
        else:
            self.label_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}

        # Select model: if None, automatically load champion model selected by training pipeline
        if model_name is None:
            metrics_path = self.root_dir / "artifacts" / "metrics.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        model_name = meta.get("best_model")
                except Exception:
                    pass
            if not model_name:
                model_name = "Linear SVM"

        self.active_model_name = model_name
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        slug = model_name.lower().replace(" ", "_")
        model_path = self.models_dir / f"{slug}.joblib"
        if not model_path.exists():
            # Try finding any available model
            for candidate in self.AVAILABLE_MODELS:
                alt_slug = candidate.lower().replace(" ", "_")
                alt_path = self.models_dir / f"{alt_slug}.joblib"
                if alt_path.exists():
                    self.active_model_name = candidate
                    return joblib.load(alt_path)
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Please train models using 'src/train.py'."
            )
        return joblib.load(model_path)

    def switch_model(self, model_name: str):
        """Switches the active classification model."""
        self.model = self._load_model(model_name)
        self.active_model_name = model_name

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies a single text review and computes probability distributions
        and key contributing tokens.
        """
        if not text or not text.strip():
            return {
                "text": text,
                "predicted_label": None,
                "sentiment": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
                "top_keywords": []
            }

        X_tfidf = self.vectorizer.transform([text])
        pred_label = int(self.model.predict(X_tfidf)[0])
        sentiment_name = self.label_map.get(pred_label, str(pred_label))

        # Probability computation with calibrated/exact probabilities
        probs = {}
        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(X_tfidf)[0]
            classes = self.model.classes_ if hasattr(self.model, "classes_") else range(len(raw_probs))
            class_prob_map = {int(cls_idx): float(p) for cls_idx, p in zip(classes, raw_probs)}
            for lbl_id in sorted(self.label_map.keys()):
                lbl_name = self.label_map[lbl_id]
                probs[lbl_name] = class_prob_map.get(lbl_id, 0.0)
            confidence = float(max(raw_probs))
        else:
            confidence = 1.0
            probs[sentiment_name] = 1.0

        # Keyword attribution (TF-IDF tokens in review)
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        row_vec = X_tfidf.toarray()[0]
        nonzero_idx = np.where(row_vec > 0)[0]

        top_keywords = []
        if len(nonzero_idx) > 0:
            sorted_idx = nonzero_idx[np.argsort(-row_vec[nonzero_idx])]
            for idx in sorted_idx[:8]:
                top_keywords.append({
                    "word": feature_names[idx],
                    "tfidf_weight": round(float(row_vec[idx]), 4)
                })

        return {
            "text": text,
            "model_used": self.active_model_name,
            "predicted_label": pred_label,
            "sentiment": sentiment_name,
            "confidence": round(confidence, 4),
            "probabilities": probs,
            "top_keywords": top_keywords
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predicts sentiment for a list of reviews."""
        return [self.predict(t) for t in texts]


if __name__ == "__main__":
    sample = "An exquisite, heart-wrenching cinematic masterpiece with stunning visuals and compelling acting."
    try:
        predictor = SentimentPredictor()
        res = predictor.predict(sample)
        print("Prediction Result:")
        print(res)
    except FileNotFoundError as err:
        print(f"[Notice] {err}")
