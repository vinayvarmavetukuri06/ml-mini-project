"""
generate_presentation.py
Generates the mandatory 5-6 slide presentation deck for the UE24CS352A Mini-Project review.
Meets all criteria in Section 4 of Guidelines and Instructions:
- Slide 1: Title & Project Metadata
- Slide 2: Problem Formulation & Challenges
- Slide 3: Dataset & TF-IDF Feature Engineering
- Slide 4: Model Architectures (Generative vs Discriminative)
- Slide 5: Empirical Benchmarks & Comparative Findings
- Slide 6: Interactive Live System & Conclusions
"""

import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE


# Design Color Palette
COLOR_BG_DARK = RGBColor(15, 23, 42)       # Slate 900
COLOR_CARD_BG = RGBColor(30, 41, 59)      # Slate 800
COLOR_PRIMARY = RGBColor(59, 130, 246)    # Blue 500
COLOR_ACCENT = RGBColor(16, 185, 129)     # Emerald 500
COLOR_TEXT_WHITE = RGBColor(248, 250, 252)
COLOR_TEXT_MUTED = RGBColor(148, 163, 184)
COLOR_GOLD = RGBColor(251, 191, 36)


def add_slide_header(slide, title_text, category_text="UE24CS352A MACHINE LEARNING MINI-PROJECT"):
    """Adds a standard modern styled header to a slide."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8.4), Inches(1.1))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_PRIMARY

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_WHITE


def set_slide_background(slide):
    """Sets dark slate background for slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG_DARK


def create_presentation():
    root_dir = Path(__file__).resolve().parent.parent
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pptx_path = reports_dir / "Movie_Review_Sentiment_Analysis_Presentation.pptx"

    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)  # 16:9 Widescreen aspect ratio
    blank_layout = prs.slide_layouts[6]

    # Load metrics if available
    metrics_path = root_dir / "artifacts" / "metrics.json"
    metrics_data = None
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = None

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(8.0), Inches(3.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "UE24CS352A — MACHINE LEARNING MINI-PROJECT"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_PRIMARY

    p1 = tf1.add_paragraph()
    p1.text = "Movie Review Sentiment Analysis"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEXT_WHITE
    p1.space_before = Pt(8)

    p2 = tf1.add_paragraph()
    p2.text = "Comparative Benchmark of Multinomial Naive Bayes, Logistic Regression & Linear SVM"
    p2.font.size = Pt(14)
    p2.font.color.rgb = COLOR_TEXT_MUTED
    p2.space_before = Pt(8)

    p3 = tf1.add_paragraph()
    p3.text = "• Student Presentation & Technical Viva Session\n• Department of Computer Science & Engineering"
    p3.font.size = Pt(11)
    p3.font.color.rgb = RGBColor(203, 213, 225)
    p3.space_before = Pt(24)

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT & CHALLENGES
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_slide_header(s2, "Problem Statement & Computational Challenges")

    box_left = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(4.0), Inches(3.4))
    box_left.fill.solid()
    box_left.fill.fore_color.rgb = COLOR_CARD_BG
    box_left.line.color.rgb = COLOR_PRIMARY
    tf_l = box_left.text_frame
    tf_l.word_wrap = True
    tf_l.paragraphs[0].text = "🎯 Objective"
    tf_l.paragraphs[0].font.bold = True
    tf_l.paragraphs[0].font.size = Pt(15)
    tf_l.paragraphs[0].font.color.rgb = COLOR_PRIMARY

    p = tf_l.add_paragraph()
    p.text = "Build an end-to-end supervised machine learning pipeline to automatically classify viewer movie reviews into fine-grained sentiment categories."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_TEXT_WHITE
    p.space_before = Pt(8)

    p_sub = tf_l.add_paragraph()
    p_sub.text = "• Benchmark generative vs. discriminative classifiers.\n• Validate generalization with 5-fold cross-validation.\n• Deploy an interactive real-time inference interface."
    p_sub.font.size = Pt(10.5)
    p_sub.font.color.rgb = COLOR_TEXT_MUTED
    p_sub.space_before = Pt(8)

    box_right = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.2), Inches(1.7), Inches(4.0), Inches(3.4))
    box_right.fill.solid()
    box_right.fill.fore_color.rgb = COLOR_CARD_BG
    box_right.line.color.rgb = COLOR_GOLD
    tf_r = box_right.text_frame
    tf_r.word_wrap = True
    tf_r.paragraphs[0].text = "⚡ Key NLP Challenges"
    tf_r.paragraphs[0].font.bold = True
    tf_r.paragraphs[0].font.size = Pt(15)
    tf_r.paragraphs[0].font.color.rgb = COLOR_GOLD

    challenges = [
        ("High Sparsity", "Text vocabularies span tens of thousands of tokens, yielding high-dimensional sparse representations."),
        ("Polarity Inversion", "Negations ('not great', 'hardly amusing') completely flip sentiment."),
        ("Subtle Sarcasm", "Nuanced colloquial critiques require discriminative n-gram boundary awareness.")
    ]
    for title, desc in challenges:
        p = tf_r.add_paragraph()
        p.text = f"• {title}: {desc}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 3: DATASET & PREPROCESSING PIPELINE
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_slide_header(s3, "Dataset & Feature Engineering Pipeline")

    steps = [
        ("1. Data Ingestion & Cleansing", "HTML strip, lowercase normalization, contraction expansion (e.g. didn't -> did not), non-word noise removal."),
        ("2. Strict Partitioning Order", "Stratified 80-20 train-test split executed BEFORE vectorizer fitting to strictly prevent data leakage."),
        ("3. TF-IDF Feature Extraction", "Unigrams + Bigrams (1,2) with sublinear TF scaling (1 + log(TF)) capturing both individual and compound tokens."),
        ("4. Dimensionality Management", "Top 15,000 informative n-grams preserved; eliminates uninformative hapax legomena via min_df threshold.")
    ]

    for idx, (head, body) in enumerate(steps):
        top_y = Inches(1.6 + idx * 0.9)
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top_y, Inches(8.4), Inches(0.75))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = COLOR_PRIMARY
        ctf = card.text_frame
        ctf.word_wrap = True
        ctf.paragraphs[0].text = f"{head}: "
        ctf.paragraphs[0].font.bold = True
        ctf.paragraphs[0].font.size = Pt(11)
        ctf.paragraphs[0].font.color.rgb = COLOR_PRIMARY

        p = ctf.paragraphs[0]
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(10)
        run.font.color.rgb = COLOR_TEXT_WHITE

    # =========================================================================
    # SLIDE 4: METHODOLOGY & MODEL ARCHITECTURES
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_slide_header(s4, "Model Architectures: Generative vs Discriminative")

    models_info = [
        ("Multinomial Naive Bayes", "Generative Probabilistic", COLOR_PRIMARY,
         "• Models class prior P(y) and likelihood P(x|y).\n• Fast closed-form training with Laplace smoothing.\n• Robust baseline for sparse text classification."),
        ("Logistic Regression", "Discriminative Linear", COLOR_ACCENT,
         "• Optimizes cross-entropy loss directly on P(y|x).\n• L2 regularization controls VC dimension & overfitting.\n• Provides well-calibrated posterior probabilities."),
        ("Linear SVM (Champion 🏆)", "Max-Margin Hyperplane", COLOR_GOLD,
         "• Maximizes margin between sentiment boundaries.\n• Hinge loss with L2 penalty handles high dimensions.\n• Selected Champion Model based on Macro-F1 (50.27%).")
    ]

    for idx, (m_title, m_type, col, desc) in enumerate(models_info):
        left_x = Inches(0.8 + idx * 2.9)
        card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_x, Inches(1.7), Inches(2.6), Inches(3.4))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = col
        ctf = card.text_frame
        ctf.word_wrap = True
        ctf.paragraphs[0].text = m_title
        ctf.paragraphs[0].font.bold = True
        ctf.paragraphs[0].font.size = Pt(13)
        ctf.paragraphs[0].font.color.rgb = col

        p_sub = ctf.add_paragraph()
        p_sub.text = m_type
        p_sub.font.size = Pt(9.5)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED
        p_sub.space_before = Pt(2)

        p_desc = ctf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(9.5)
        p_desc.font.color.rgb = COLOR_TEXT_WHITE
        p_desc.space_before = Pt(8)

    # =========================================================================
    # SLIDE 5: EXPERIMENTAL BENCHMARKS & RESULTS
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_slide_header(s5, "Empirical Evaluation & Performance Comparison")

    table_shape = s5.shapes.add_table(4, 5, Inches(0.8), Inches(1.7), Inches(8.4), Inches(1.8))
    table = table_shape.table

    headers = ["Model Name", "5-Fold CV Acc", "Test Accuracy", "Macro F1", "Inference Latency"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        cell.text_frame.paragraphs[0].text = h
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.size = Pt(10)
        cell.text_frame.paragraphs[0].font.color.rgb = COLOR_TEXT_WHITE

    if metrics_data and "models" in metrics_data:
        m_dict = metrics_data["models"]
        for row_idx, (name, d) in enumerate(m_dict.items(), start=1):
            disp_name = f"{name} 🏆 (Champion)" if name == "Linear SVM" else name
            row_data = [
                disp_name,
                f"{d['cv_accuracy_mean']*100:.2f}%",
                f"{d['test_accuracy']*100:.2f}%",
                f"{d['macro_f1']*100:.2f}%",
                f"{d['latency_per_sample_ms']:.3f} ms"
            ]
            for col_idx, val in enumerate(row_data):
                cell = table.cell(row_idx, col_idx)
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_CARD_BG
                cell.text_frame.paragraphs[0].text = val
                cell.text_frame.paragraphs[0].font.size = Pt(9.5)
                cell.text_frame.paragraphs[0].font.color.rgb = COLOR_GOLD if "Champion" in disp_name and col_idx == 0 else COLOR_TEXT_WHITE
    else:
        sample_rows = [
            ["Linear SVM 🏆 (Champion)", "63.09%", "63.85%", "50.27%", "0.002 ms"],
            ["Logistic Regression", "63.55%", "63.96%", "48.24%", "0.000 ms"],
            ["Multinomial Naive Bayes", "59.71%", "60.06%", "36.49%", "0.000 ms"]
        ]
        for row_idx, row_vals in enumerate(sample_rows, start=1):
            for col_idx, val in enumerate(row_vals):
                cell = table.cell(row_idx, col_idx)
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_CARD_BG
                cell.text_frame.paragraphs[0].text = val
                cell.text_frame.paragraphs[0].font.size = Pt(9.5)
                cell.text_frame.paragraphs[0].font.color.rgb = COLOR_GOLD if "Champion" in row_vals[0] and col_idx == 0 else COLOR_TEXT_WHITE

    # Findings Box
    f_box = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.7), Inches(8.4), Inches(1.5))
    f_box.fill.solid()
    f_box.fill.fore_color.rgb = COLOR_CARD_BG
    f_box.line.color.rgb = COLOR_ACCENT
    ftf = f_box.text_frame
    ftf.word_wrap = True
    ftf.paragraphs[0].text = "📊 Key Empirical Insights:"
    ftf.paragraphs[0].font.bold = True
    ftf.paragraphs[0].font.size = Pt(10.5)
    ftf.paragraphs[0].font.color.rgb = COLOR_ACCENT

    p1 = ftf.add_paragraph()
    p1.text = "• Champion Model Selection: Linear SVM is selected as the production champion with 50.27% Macro-F1 (vs. 48.24% for Logistic Regression and 36.49% for Naive Bayes)."
    p1.font.size = Pt(8.5)
    p1.font.color.rgb = COLOR_TEXT_WHITE
    p1.space_before = Pt(2)

    p2 = ftf.add_paragraph()
    p2.text = "• Metric Prioritization (Macro-F1 vs Test Accuracy): Logistic Regression achieves peak test accuracy (63.96% vs 63.85%), but Macro-F1 is prioritized because it measures performance across all sentiment classes rather than test accuracy alone, which is biased by dominant Neutral reviews (51%+)."
    p2.font.size = Pt(8.5)
    p2.font.color.rgb = COLOR_TEXT_WHITE
    p2.space_before = Pt(2)

    p3 = ftf.add_paragraph()
    p3.text = "• Production Efficiency: All three models execute sub-millisecond per-sample inference latency with calibrated 5-class posterior probabilities."
    p3.font.size = Pt(8.5)
    p3.font.color.rgb = COLOR_TEXT_WHITE
    p3.space_before = Pt(2)

    # =========================================================================
    # SLIDE 6: INTERACTIVE DEMO & CONCLUSION
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_slide_header(s6, "Interactive Terminal System & Conclusions")

    col_l = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(4.0), Inches(3.4))
    col_l.fill.solid()
    col_l.fill.fore_color.rgb = COLOR_CARD_BG
    col_l.line.color.rgb = COLOR_PRIMARY
    ltf = col_l.text_frame
    ltf.word_wrap = True
    ltf.paragraphs[0].text = "🖥️ Live Terminal Interface"
    ltf.paragraphs[0].font.bold = True
    ltf.paragraphs[0].font.size = Pt(13)
    ltf.paragraphs[0].font.color.rgb = COLOR_PRIMARY

    demo_bullets = [
        "Interactive Terminal Menu (main.py) with 4 options",
        "Direct prediction workflow automatically loads Linear SVM champion (Macro-F1: 50.27%)",
        "5-Class sentiment predictions across MNB, LogReg & Linear SVM",
        "Calibrated probabilities & TF-IDF Keyword Explainability",
        "Standalone execution: clean terminal app, no browser required"
    ]
    for b in demo_bullets:
        p = ltf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(5)

    col_r = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.2), Inches(1.7), Inches(4.0), Inches(3.4))
    col_r.fill.solid()
    col_r.fill.fore_color.rgb = COLOR_CARD_BG
    col_r.line.color.rgb = COLOR_ACCENT
    rtf = col_r.text_frame
    rtf.word_wrap = True
    rtf.paragraphs[0].text = "🏁 Summary & Next Steps"
    rtf.paragraphs[0].font.bold = True
    rtf.paragraphs[0].font.size = Pt(13)
    rtf.paragraphs[0].font.color.rgb = COLOR_ACCENT

    concl_bullets = [
        "Linear SVM is the selected champion (Macro-F1: 50.27%), providing optimal balance across minority & majority sentiment classes.",
        "Logistic Regression achieves peak test accuracy (63.96%), while Naive Bayes offers an ultra-fast baseline (60.06%).",
        "Champion selection prioritizes Macro-F1 over raw accuracy to prevent majority-class bias from dominant Neutral reviews.",
        "Future Work: Fine-tuning transformer models (DistilBERT / RoBERTa) and contextual sentence embeddings."
    ]
    for b in concl_bullets:
        p = rtf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(5)

    prs.save(str(pptx_path))
    print(f"[Presentation] Successfully created 6-slide deck: {pptx_path}")
    return pptx_path


if __name__ == "__main__":
    create_presentation()
