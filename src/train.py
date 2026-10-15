"""
train.py
End-to-end comparative machine learning training pipeline for Sentiment Analysis.
Compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM with TF-IDF features.
Adheres strictly to ML Best Practices: separate train/test splits, 5-fold cross-validation,
rigorous metric evaluation, and artifact serialization.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Relative imports
try:
    from src.data_loader import load_dataset
    from src.preprocessor import create_vectorizer
except ModuleNotFoundError:
    # Handle direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_dataset
    from src.preprocessor import create_vectorizer


def train_and_evaluate_models(dataset_path: str = None) -> Dict[str, Any]:
    """
    Executes the training and benchmarking workflow.
    """
    root_dir = Path(__file__).resolve().parent.parent
    artifacts_dir = root_dir / "artifacts"
    models_dir = artifacts_dir / "models"
    plots_dir = artifacts_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MOVIE REVIEW SENTIMENT ANALYSIS - ML PIPELINE")
    print("  Models: Multinomial Naive Bayes vs. Logistic Regression vs. Linear SVM")
    print("=" * 70)

    # 1. Load Data
    df, meta = load_dataset(dataset_path)
    X = np.array(df["text"].tolist(), dtype=object)
    y = np.array(df["label"].tolist(), dtype=int)
    
    # Save label mappings
    unique_labels = sorted(np.unique(y).tolist())
    label_id_to_name = {}
    for idx, row in df.drop_duplicates(subset=["label"]).iterrows():
        label_id_to_name[int(row["label"])] = str(row["label_name"])
    
    joblib.dump(label_id_to_name, models_dir / "label_map.joblib")

    # 2. Train-Test Split (Strict ordering: split BEFORE feature extraction)
    print("\n[1/5] Splitting data into 80% train and 20% test (Stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Training samples: {len(X_train_raw):,}")
    print(f"      Testing samples:  {len(X_test_raw):,}")

    # 3. Fit TF-IDF Vectorizer ONLY on training set
    print("\n[2/5] Extracting TF-IDF Features (unigrams + bigrams, sublinear TF)...")
    vectorizer = create_vectorizer(ngram_range=(1, 2), max_features=15000)
    t0 = time.time()
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    feat_time = time.time() - t0
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(f"      Vocabulary size: {X_train_tfidf.shape[1]:,} features")
    print(f"      Feature extraction time: {feat_time:.2f}s")

    # Save vectorizer
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.joblib")

    # 4. Define candidate models
    # Note: We calibrate LinearSVC so it outputs well-calibrated class probabilities for Streamlit UI
    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, dual="auto", random_state=42, max_iter=2000), cv=3)
    }

    # 5. Model Evaluation and Cross-Validation
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[3/5] Benchmarking Models with 5-Fold Cross-Validation & Test Holdout...")

    for model_name, model in candidate_models.items():
        print(f"\n---> Training and evaluating: {model_name}")

        # 5-fold CV on training set
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"     5-Fold CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

        # Fit on full training set
        t_train_start = time.time()
        model.fit(X_train_tfidf, y_train)
        train_duration = time.time() - t_train_start
        print(f"     Train time: {train_duration:.2f}s")

        # Measure test latency and predictions
        t_infer_start = time.time()
        y_pred = model.predict(X_test_tfidf)
        infer_duration = time.time() - t_infer_start
        latency_per_sample_ms = (infer_duration / len(X_test_raw)) * 1000.0

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels).tolist()
        report_dict = classification_report(
            y_test, y_pred, labels=unique_labels,
            target_names=[label_id_to_name[i] for i in unique_labels],
            output_dict=True, zero_division=0
        )

        print(f"     Test Accuracy:      {acc * 100:.2f}%")
        print(f"     Test Macro F1:      {f1_macro * 100:.2f}%")
        print(f"     Inference Latency:  {latency_per_sample_ms:.3f} ms/sample")

        # Model file slug
        slug = model_name.lower().replace(" ", "_")
        joblib.dump(model, models_dir / f"{slug}.joblib")

        results[model_name] = {
            "model_name": model_name,
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
            "test_accuracy": float(acc),
            "macro_precision": float(prec_macro),
            "macro_recall": float(rec_macro),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_wt),
            "train_duration_sec": float(train_duration),
            "latency_per_sample_ms": float(latency_per_sample_ms),
            "confusion_matrix": cm,
            "classification_report": report_dict,
            "artifact_path": str(models_dir / f"{slug}.joblib")
        }

    # Best model selection by Macro F1
    best_model_name = max(results.keys(), key=lambda k: results[k]["macro_f1"])
    print("\n[4/5] Champion Model Selection:")
    print(f"      Best Model: {best_model_name} (Macro F1: {results[best_model_name]['macro_f1']*100:.2f}%)")

    # Save summary metadata
    summary = {
        "dataset_metadata": meta,
        "labels": {str(k): v for k, v in label_id_to_name.items()},
        "best_model": best_model_name,
        "models": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"      Saved metrics to: {metrics_path}")

    # 6. Generate evaluation plots and reports
    print("\n[5/5] Generating evaluation charts and visual reports...")
    try:
        from src.evaluate import generate_all_evaluation_plots
        generate_all_evaluation_plots(summary)
    except Exception as exc:
        print(f"[Warning] Evaluation plot generation failed: {exc}")

    print("\n" + "=" * 70)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    custom_dataset = sys.argv[1] if len(sys.argv) > 1 else None
    train_and_evaluate_models(custom_dataset)
