"""
test_evaluation.py
Tests evaluation metrics structure, metric values, champion model selection criterion,
and markdown report generation logic.
"""

from src.evaluate import print_markdown_report


def test_evaluation_metrics_integrity(metrics_data):
    """Verifies that artifacts/metrics.json contains valid evaluation stats for all 3 models."""
    assert "best_model" in metrics_data
    assert "models" in metrics_data

    models = metrics_data["models"]
    for expected_model in ["Linear SVM", "Logistic Regression", "Multinomial Naive Bayes"]:
        assert expected_model in models, f"Model {expected_model} missing from evaluation metrics"
        m_data = models[expected_model]

        assert 0.0 < m_data["test_accuracy"] <= 1.0
        assert 0.0 < m_data["macro_f1"] <= 1.0
        assert 0.0 < m_data["weighted_f1"] <= 1.0
        assert 0.0 < m_data["cv_accuracy_mean"] <= 1.0

        cm = m_data.get("confusion_matrix")
        assert isinstance(cm, list)
        assert len(cm) == 5, f"Confusion matrix for {expected_model} must be 5x5"
        for row in cm:
            assert len(row) == 5


def test_champion_model_selection_criterion(metrics_data):
    """Verifies that the champion model corresponds to the model with highest macro F1."""
    models = metrics_data["models"]
    best_name = metrics_data["best_model"]

    best_macro_f1 = max(m["macro_f1"] for m in models.values())
    assert models[best_name]["macro_f1"] == best_macro_f1, (
        f"Champion model '{best_name}' does not have the highest macro F1 ({best_macro_f1})"
    )


def test_print_markdown_report_formatting(metrics_data, capsys):
    """Verifies print_markdown_report formats a valid markdown summary table."""
    print_markdown_report(metrics_data)
    captured = capsys.readouterr().out

    assert "### Benchmark Summary Table" in captured
    assert "| Model | 5-Fold CV Acc | Test Acc | Macro F1 |" in captured
    assert "Linear SVM" in captured
    assert "Logistic Regression" in captured
    assert "Multinomial Naive Bayes" in captured
    assert f"**Best Model Selected**: {metrics_data['best_model']}" in captured
