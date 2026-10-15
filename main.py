"""
main.py
Interactive terminal-based application for Movie Review Sentiment Analysis.
UE24CS352A - Machine Learning Mini-Project
"""

import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows consoles if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import find_dataset_file
from src.predict import SentimentPredictor


BANNER = r"""
================================================================================
  [MOVIE REVIEW SENTIMENT ANALYSIS - TERMINAL SYSTEM]
  UE24CS352A Machine Learning Mini-Project
  Multinomial Naive Bayes | Logistic Regression | Linear SVM (5 Classes)
================================================================================
"""

MENU = """
--------------------------------------------------------------------------------
MAIN MENU:
  [1] Train and compare ML models
  [2] Predict sentiment for a custom movie review
  [3] Evaluate saved models and generate deliverables (PDF & PPTX)
  [4] Exit
--------------------------------------------------------------------------------
"""


def format_bar(pct: float, width: int = 24) -> str:
    """Formats a clean ASCII progress bar."""
    filled = int(round((pct / 100.0) * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


def option_train():
    """Option 1: Train and compare models."""
    print("\n" + "=" * 70)
    print("  STARTING MODEL TRAINING & 5-FOLD CROSS-VALIDATION BENCHMARK")
    print("=" * 70)

    dataset_path = find_dataset_file()
    if dataset_path is None:
        print("[Error] No dataset found in data/. Please provide train.tsv in data/.")
        return

    print(f"[Dataset] Target file: {dataset_path.name}")
    try:
        from src.train import train_and_evaluate_models
        summary = train_and_evaluate_models(str(dataset_path))
        print("\n[Success] Training completed successfully!")
        print_evaluation_summary(summary)
    except Exception as exc:
        print(f"[Error] Training failed: {exc}")
        import traceback
        traceback.print_exc()


def print_evaluation_summary(summary: dict):
    """Displays benchmark comparison table."""
    models_data = summary.get("models", {})
    if not models_data:
        print("[Notice] No evaluation results found in metrics.")
        return

    print("\n" + "-" * 78)
    print(f"{'MODEL':<25} | {'CV ACC (5-Fold)':<16} | {'TEST ACC':<9} | {'MACRO F1':<9} | {'LATENCY':<10}")
    print("-" * 78)

    for name, data in models_data.items():
        cv_str = f"{data['cv_accuracy_mean']*100:.2f}% (+/-{data['cv_accuracy_std']*100:.2f}%)"
        test_acc = f"{data['test_accuracy']*100:.2f}%"
        macro_f1 = f"{data['macro_f1']*100:.2f}%"
        latency = f"{data['latency_per_sample_ms']:.3f} ms"
        print(f"{name:<25} | {cv_str:<16} | {test_acc:<9} | {macro_f1:<9} | {latency:<10}")

    print("-" * 78)
    print(f"[CHAMPION] Best Model Selected: {summary.get('best_model')}\n")


def option_predict():
    """Option 2: Predict sentiment for custom review."""
    print("\n" + "=" * 70)
    print("  SENTIMENT PREDICTION ENGINE")
    print("=" * 70)

    models_dir = ROOT_DIR / "artifacts" / "models"
    if not (models_dir / "tfidf_vectorizer.joblib").exists():
        print("\n[Warning] Model artifacts not found. Please train models first using Option [1].")
        return

    # Automatically load the champion model selected by training pipeline
    metrics_file = ROOT_DIR / "artifacts" / "metrics.json"
    champion_model = None
    champion_macro_f1 = None
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                champion_model = meta.get("best_model")
                if champion_model and champion_model in meta.get("models", {}):
                    champion_macro_f1 = meta["models"][champion_model].get("macro_f1")
        except Exception:
            pass

    try:
        predictor = SentimentPredictor(model_name=champion_model)
        f1_str = f" (Macro F1: {champion_macro_f1*100:.2f}%)" if champion_macro_f1 else ""
        print(f"[Champion Model] Automatically loaded '{predictor.active_model_name}' (Criterion: Highest Macro F1-Score{f1_str}).")
    except Exception as exc:
        print(f"[Error] Could not initialize predictor: {exc}")
        return

    print("Type your review text below, or type 'sample' to pick a preset review.")

    sample_reviews = [
        ("Very Positive", "An exquisite, heart-wrenching cinematic masterpiece with breathtaking visuals and sublime acting."),
        ("Positive", "A charming, thoroughly enjoyable feel-good comedy with great comic timing."),
        ("Neutral", "A moderately competent procedural thriller that follows standard genre conventions without surprises."),
        ("Negative", "The screenplay is disjointed, dialogue is awkward, and the pacing drags painfully."),
        ("Very Negative", "A complete and utter catastrophe. Abysmal acting, hideous editing, and unwatchable nonsense.")
    ]

    while True:
        try:
            user_input = input("\nEnter your movie review (or type 'back' to return): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nReturning to main menu.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["back", "exit", "quit", "q"]:
            break

        if user_input.lower() == "sample":
            print("\nSelect a sample review to test:")
            for s_idx, (cat, s_text) in enumerate(sample_reviews, start=1):
                print(f"  [{s_idx}] ({cat}): \"{s_text[:60]}...\"")
            s_choice = input("Pick sample (1-5): ").strip()
            if s_choice in ["1", "2", "3", "4", "5"]:
                user_input = sample_reviews[int(s_choice) - 1][1]
                print(f"\nEvaluating Sample: \"{user_input}\"")
            else:
                continue

        # Predict
        res = predictor.predict(user_input)
        sentiment = res["sentiment"]
        conf = res["confidence"] * 100
        probs = res.get("probabilities", {})
        keywords = res.get("top_keywords", [])

        # Display result card
        print("\n" + "-" * 60)
        print("  PREDICTION RESULT")
        print("-" * 60)
        print(f"  Model Used:       {res['model_used']}")
        print(f"  Predicted Class:  {sentiment.upper()} (Class {res['predicted_label']})")
        print(f"  Model Confidence: {conf:.1f}%")
        print("-" * 60)

        if probs:
            print("  Probability Distribution:")
            for class_name, prob_val in probs.items():
                pct = prob_val * 100.0
                marker = " *" if class_name == sentiment else "  "
                print(f"    {class_name:<15}: {format_bar(pct)}{marker}")
            print("-" * 60)

        if keywords:
            kw_str = ", ".join([f"'{k['word']}' ({k['tfidf_weight']})" for k in keywords[:5]])
            print(f"  Top Contributing TF-IDF Tokens:\n    {kw_str}")
            print("-" * 60)


def option_evaluate():
    """Option 3: Evaluate saved models and regenerate reports."""
    print("\n" + "=" * 70)
    print("  MODEL EVALUATION & DELIVERABLES GENERATION")
    print("=" * 70)

    metrics_file = ROOT_DIR / "artifacts" / "metrics.json"
    if not metrics_file.exists():
        print("\n[Warning] No metrics found at artifacts/metrics.json. Please run Option [1] to train models first.")
        return

    with open(metrics_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 1. Summary comparison table
    print_evaluation_summary(summary)

    # 2. Detailed Per-Class Breakdown
    print("\nDetailed Per-Class Classification Reports:")
    for m_name, m_data in summary.get("models", {}).items():
        print(f"\n--- Model: {m_name} ---")
        rep = m_data.get("classification_report", {})
        print(f"{'Class':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        print("-" * 62)
        for label, scores in rep.items():
            if isinstance(scores, dict):
                prec = f"{scores.get('precision', 0)*100:.1f}%"
                rec = f"{scores.get('recall', 0)*100:.1f}%"
                f1 = f"{scores.get('f1-score', 0)*100:.1f}%"
                supp = int(scores.get('support', 0))
                print(f"{label:<16} | {prec:<10} | {rec:<10} | {f1:<10} | {supp:<8}")
        print("-" * 62)

    # 3. Generate Evaluation Visuals
    print("\n[Visuals] Regenerating evaluation plots...")
    try:
        from src.evaluate import generate_all_evaluation_plots
        generate_all_evaluation_plots(summary)
        print("  [OK] Saved plots to: artifacts/plots/")
    except Exception as exc:
        print(f"  [Warning] Plot generation failed: {exc}")

    # 4. Generate Two-Page PDF Report
    print("\n[Reports] Generating Two-Page PDF Write-up...")
    try:
        from reports.generate_pdf_report import create_two_page_report
        pdf_path = create_two_page_report()
        print(f"  [OK] Compiled PDF Report: {pdf_path}")
    except Exception as exc:
        print(f"  [Warning] PDF generation failed: {exc}")

    # 5. Generate PowerPoint Deck
    print("\n[Reports] Generating 6-Slide PowerPoint Deck...")
    try:
        from reports.generate_presentation import create_presentation
        pptx_path = create_presentation()
        print(f"  [OK] Compiled Presentation: {pptx_path}")
    except Exception as exc:
        print(f"  [Warning] Presentation generation failed: {exc}")

    print("\n" + "=" * 70)
    print("  EVALUATION & DELIVERABLES READY")
    print("=" * 70)


def main():
    """Main program entry point."""
    print(BANNER)

    # Pre-check dataset
    ds = find_dataset_file()
    if ds:
        print(f"Found active dataset: {ds.name} ({ds.stat().st_size:,} bytes)")
    else:
        print("Warning: No dataset found in data/. Place train.tsv into data/.")

    while True:
        print(MENU)
        try:
            choice = input("Enter option (1-4): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            sys.exit(0)

        if choice == "1":
            option_train()
        elif choice == "2":
            option_predict()
        elif choice == "3":
            option_evaluate()
        elif choice == "4":
            print("\nExiting Movie Review Sentiment Analysis System. Goodbye!")
            sys.exit(0)
        else:
            print("\n[Invalid Option] Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
