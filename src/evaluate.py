"""
evaluate.py
Evaluation and reporting module.
Produces publication-grade comparison plots, confusion matrices, and markdown summary tables.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_all_evaluation_plots(summary: Optional[Dict[str, Any]] = None):
    """
    Renders and saves comprehensive evaluation visuals:
    - Model comparison metrics bar chart
    - Confusion matrix heatmaps for all 3 models
    - Training duration and inference latency trade-offs
    - Class balance distribution
    """
    root_dir = Path(__file__).resolve().parent.parent
    artifacts_dir = root_dir / "artifacts"
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if summary is None:
        metrics_file = artifacts_dir / "metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file {metrics_file} not found. Please run src/train.py first.")
        with open(metrics_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

    models_data = summary.get("models", {})
    if not models_data:
        print("[Evaluate] No model results found in summary.")
        return

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.titlesize": 15
    })

    model_names = list(models_data.keys())
    test_accs = [models_data[m]["test_accuracy"] * 100 for m in model_names]
    cv_accs = [models_data[m]["cv_accuracy_mean"] * 100 for m in model_names]
    macro_f1s = [models_data[m]["macro_f1"] * 100 for m in model_names]
    weighted_f1s = [models_data[m]["weighted_f1"] * 100 for m in model_names]
    latencies = [models_data[m]["latency_per_sample_ms"] for m in model_names]
    train_times = [models_data[m]["train_duration_sec"] for m in model_names]

    # 1. Bar Chart: Comprehensive Metric Comparison
    x = np.arange(len(model_names))
    width = 0.20

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    rects1 = ax.bar(x - 1.5 * width, cv_accs, width, label="5-Fold CV Acc (%)", color="#4285F4")
    rects2 = ax.bar(x - 0.5 * width, test_accs, width, label="Test Acc (%)", color="#34A853")
    rects3 = ax.bar(x + 0.5 * width, macro_f1s, width, label="Macro F1 (%)", color="#FBBC05")
    rects4 = ax.bar(x + 1.5 * width, weighted_f1s, width, label="Weighted F1 (%)", color="#EA4335")

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison: Naive Bayes vs. Logistic Regression vs. Linear SVM")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", loc="lower right")
    ax.set_ylim(0, 105)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    comparison_plot_path = plots_dir / "model_comparison_bar.png"
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"[Evaluate] Saved: {comparison_plot_path}")

    # 2. Confusion Matrices Subplots
    raw_labels = summary.get("labels", {})
    label_keys_sorted = sorted([int(k) for k in raw_labels.keys()])
    display_names = [raw_labels[str(k)] for k in label_keys_sorted]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=300)
    for idx, (m_name, ax) in enumerate(zip(model_names, axes)):
        cm = np.array(models_data[m_name]["confusion_matrix"])
        # Row-normalize to percentages
        with np.errstate(all="ignore"):
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(
            cm_norm, annot=True, fmt=".1%", annot_kws={"size": 8.5},
            cmap="Blues" if idx == 0 else ("Greens" if idx == 1 else "Purples"),
            xticklabels=display_names, yticklabels=display_names, ax=ax, cbar=False
        )
        ax.set_title(f"{m_name}\n(Acc: {test_accs[idx]:.1f}%, F1: {macro_f1s[idx]:.1f}%)", fontweight="bold")
        ax.set_xlabel("Predicted Sentiment")
        ax.set_ylabel("True Sentiment" if idx == 0 else "")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.suptitle("5×5 Confusion Matrices (Normalized across True Sentiments)", y=1.02, fontsize=16)
    plt.tight_layout()
    cm_plot_path = plots_dir / "confusion_matrices.png"
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"[Evaluate] Saved: {cm_plot_path}")

    # 3. Efficiency & Latency Trade-Off Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    colors = ["#4285F4", "#34A853", "#EA4335"]

    # Training time
    bars1 = ax1.bar(model_names, train_times, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Training Time (seconds)")
    ax1.set_title("Training Duration")
    for bar in bars1:
        ax1.annotate(f"{bar.get_height():.2f}s",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    # Inference latency
    bars2 = ax2.bar(model_names, latencies, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_ylabel("Latency per Sample (ms)")
    ax2.set_title("Inference Latency")
    for bar in bars2:
        ax2.annotate(f"{bar.get_height():.3f} ms",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    plt.tight_layout()
    latency_plot_path = plots_dir / "latency_and_efficiency.png"
    plt.savefig(latency_plot_path)
    plt.close()
    print(f"[Evaluate] Saved: {latency_plot_path}")

    # 4. Class Distribution Plot
    dist = summary.get("dataset_metadata", {}).get("class_distribution", {})
    if dist:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        classes = list(dist.keys())
        counts = list(dist.values())
        sns.barplot(x=classes, y=counts, hue=classes, palette="viridis", legend=False, ax=ax)
        ax.set_title("Sentiment Class Distribution in Dataset", fontweight="bold")
        ax.set_ylabel("Number of Samples")
        for i, v in enumerate(counts):
            ax.text(i, v + (max(counts) * 0.01), f"{v:,}", ha="center", fontweight="bold")
        plt.tight_layout()
        dist_plot_path = plots_dir / "class_distribution.png"
        plt.savefig(dist_plot_path)
        plt.close()
        print(f"[Evaluate] Saved: {dist_plot_path}")


def print_markdown_report(summary: Dict[str, Any]):
    """Generates markdown summary table."""
    models_data = summary.get("models", {})
    print("\n### Benchmark Summary Table\n")
    print("| Model | 5-Fold CV Acc | Test Acc | Macro F1 | Weighted F1 | Train Time | Latency/Sample |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, data in models_data.items():
        print(f"| **{name}** | {data['cv_accuracy_mean']*100:.2f}% (±{data['cv_accuracy_std']*100:.2f}%) | "
              f"{data['test_accuracy']*100:.2f}% | {data['macro_f1']*100:.2f}% | {data['weighted_f1']*100:.2f}% | "
              f"{data['train_duration_sec']:.2f}s | {data['latency_per_sample_ms']:.3f} ms |")
    print(f"\n**Best Model Selected**: {summary.get('best_model')}\n")


if __name__ == "__main__":
    generate_all_evaluation_plots()
