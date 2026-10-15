"""
app.py
Interactive Streamlit Web Application for Movie Review Sentiment Analysis.
Features live text classification, multi-model comparison, TF-IDF explainability,
and academic mini-project deliverables display.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.predict import SentimentPredictor
    from src.data_loader import find_dataset_file
except ImportError:
    pass

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Movie Review Sentiment Analysis | ML Mini-Project",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN STYLESHEET (GLASSMORPHISM & DARK THEME) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: radial-gradient(circle at 15% 15%, #1a1e29 0%, #0d1117 100%);
        color: #e6edf3;
    }

    /* Gradient Header */
    .hero-title {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
    }

    .sentiment-banner-pos {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #34d399;
    }

    .sentiment-banner-neg {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.3) 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #f87171;
    }

    .sentiment-banner-neu {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(180, 83, 9, 0.3) 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #fbbf24;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-primary { background: #3b82f6; color: white; }
    .badge-secondary { background: #475569; color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)


# --- CACHED RESOURCES ---
@st.cache_resource
def get_predictor(model_name: str):
    try:
        return SentimentPredictor(model_name=model_name)
    except Exception:
        return None

@st.cache_data
def load_metrics():
    metrics_path = ROOT_DIR / "artifacts" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### 🎬 Project Controls")
    st.markdown("**Course**: UE24CS352A - Machine Learning")
    st.markdown("**Domain**: Natural Language Processing")

    st.markdown("---")
    st.markdown("#### 🧠 Classification Model")
    selected_model = st.selectbox(
        "Active Classifier",
        ["Logistic Regression", "Linear SVM", "Multinomial Naive Bayes"],
        index=0,
        help="Select the ML model to evaluate and use for live inference."
    )

    predictor = get_predictor(selected_model)
    if predictor:
        st.success(f"Loaded: **{selected_model}**", icon="✅")
    else:
        st.warning("Model artifacts not yet generated.", icon="⚠️")
        st.caption("Run `python src/train.py` after adding the dataset.")

    st.markdown("---")
    dataset_file = find_dataset_file()
    if dataset_file:
        st.info(f"📂 Dataset found:\n`{dataset_file.name}`", icon="📊")
    else:
        st.error("❌ Dataset missing in `data/`", icon="📁")
        st.caption("Place `train.tsv` or `movie_reviews.csv` in `data/`.")

    st.markdown("---")
    st.caption("Crafted with Scikit-Learn, Streamlit & TF-IDF.")


# --- MAIN HEADER ---
st.markdown('<div class="hero-title">Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Comparative Evaluation of Multinomial Naive Bayes, Logistic Regression & Linear SVM with TF-IDF Features</div>',
    unsafe_allow_html=True
)


# --- TABS ---
tab_predict, tab_compare, tab_deliverables, tab_dataset = st.tabs([
    "🔮 Live Review Inference",
    "📈 Model Comparison & Benchmark",
    "📄 Mini-Project Deliverables (PDF & Slides)",
    "🗂️ Dataset & Setup Guide"
])


# ==============================================================================
# TAB 1: LIVE PREDICTION
# ==============================================================================
with tab_predict:
    st.markdown("### Test the Model on Custom Movie Reviews")
    st.write("Type or paste any movie review below, or choose one of our sample reviews to test generalization:")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    sample_text = ""
    if col_btn1.button("🌟 Rave Masterpiece"):
        sample_text = "An exquisite, heart-wrenching cinematic masterpiece with breathtaking visuals, incredible depth, and sublime acting."
    elif col_btn2.button("👎 Total Disaster"):
        sample_text = "A complete waste of time. Terribly written, abysmal acting, painfully sluggish pacing, and utter nonsense."
    elif col_btn3.button("⚖️ Mixed / Nuanced"):
        sample_text = "The visual effects and cinematography are dazzling, but the screenplay is disjointed and emotionally hollow."
    elif col_btn4.button("😐 Flat & Mediocre"):
        sample_text = "Neither notably offensive nor particularly engaging; an entirely ordinary and forgettable popcorn flick."

    review_input = st.text_area(
        "Enter Movie Review:",
        value=sample_text if sample_text else "",
        placeholder="e.g., 'Christopher Nolan delivers a magnificent spectacle with breathtaking pacing and brilliant sound design...'",
        height=130
    )

    col_act1, col_act2 = st.columns([1, 4])
    predict_clicked = col_act1.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    if (predict_clicked or sample_text) and review_input.strip():
        if predictor is None:
            st.error("Please run the training pipeline first (`python src/train.py`) to build model artifacts.")
        else:
            with st.spinner(f"Analyzing with {selected_model}..."):
                res = predictor.predict(review_input)

            sentiment = res["sentiment"].lower()
            conf_pct = res["confidence"] * 100

            st.markdown("<br>", unsafe_allow_html=True)
            col_res_left, col_res_right = st.columns([1.2, 1.8])

            with col_res_left:
                st.markdown("#### Prediction Verdict")
                if "pos" in sentiment:
                    st.markdown(f"""
                    <div class="sentiment-banner-pos">
                        <h2 style="margin:0; font-size:1.8rem;">👍 {res['sentiment']}</h2>
                        <p style="margin:0; font-size:0.95rem;">Model Confidence: <strong>{conf_pct:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                elif "neg" in sentiment:
                    st.markdown(f"""
                    <div class="sentiment-banner-neg">
                        <h2 style="margin:0; font-size:1.8rem;">👎 {res['sentiment']}</h2>
                        <p style="margin:0; font-size:0.95rem;">Model Confidence: <strong>{conf_pct:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="sentiment-banner-neu">
                        <h2 style="margin:0; font-size:1.8rem;">⚖️ {res['sentiment']}</h2>
                        <p style="margin:0; font-size:0.95rem;">Model Confidence: <strong>{conf_pct:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"<p style='margin-top:0.8rem; color:#94a3b8;'>Classifier: <code>{res['model_used']}</code></p>", unsafe_allow_html=True)

            with col_res_right:
                st.markdown("#### Probability Distribution")
                probs = res.get("probabilities", {})
                if probs:
                    prob_df = pd.DataFrame(list(probs.items()), columns=["Sentiment Class", "Probability"])
                    prob_df["Score (%)"] = prob_df["Probability"] * 100
                    st.bar_chart(prob_df.set_index("Sentiment Class")["Score (%)"], color="#3b82f6")
                else:
                    st.info("Probability output not supported for raw decision function.")

            # Keyword explainability
            st.markdown("---")
            st.markdown("#### 🔍 TF-IDF Explainability & Key Discriminative Tokens")
            st.caption("These extracted n-grams contributed most strongly to the model's feature representation for this review:")
            keywords = res.get("top_keywords", [])
            if keywords:
                cols = st.columns(min(len(keywords), 6))
                for idx, kw in enumerate(keywords[:6]):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align:center;">
                            <span style="font-weight:700; color:#60a5fa; font-size:1.1rem;">{kw['word']}</span><br>
                            <span style="font-size:0.8rem; color:#94a3b8;">TF-IDF: {kw['tfidf_weight']}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.caption("No significant vocabulary tokens recognized in vocabulary dictionary.")


# ==============================================================================
# TAB 2: MODEL COMPARISON & BENCHMARK
# ==============================================================================
with tab_compare:
    st.markdown("### Comparative Evaluation of Algorithms")
    metrics_data = load_metrics()

    if metrics_data is None:
        st.warning("⚠️ No benchmark results found yet. Execute `python src/train.py` to generate rigorous benchmarks on your dataset.")
    else:
        st.success(f"🏆 Champion Model: **{metrics_data.get('best_model', 'N/A')}** based on holdout Macro F1-score.")

        models_dict = metrics_data.get("models", {})

        # Top Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        best_name = metrics_data.get("best_model")
        best_stats = models_dict.get(best_name, {})

        with col_m1:
            st.metric("Test Accuracy", f"{best_stats.get('test_accuracy', 0)*100:.2f}%", f"{best_name}")
        with col_m2:
            st.metric("5-Fold CV Accuracy", f"{best_stats.get('cv_accuracy_mean', 0)*100:.2f}%", f"±{best_stats.get('cv_accuracy_std', 0)*100:.2f}%")
        with col_m3:
            st.metric("Macro F1-Score", f"{best_stats.get('macro_f1', 0)*100:.2f}%", "Balanced metric")
        with col_m4:
            st.metric("Inference Latency", f"{best_stats.get('latency_per_sample_ms', 0):.3f} ms", "per review")

        st.markdown("---")
        st.markdown("#### Benchmark Summary Table")
        table_rows = []
        for name, d in models_dict.items():
            table_rows.append({
                "Model": name,
                "5-Fold CV Acc (%)": f"{d['cv_accuracy_mean']*100:.2f}% (±{d['cv_accuracy_std']*100:.2f}%)",
                "Test Acc (%)": f"{d['test_accuracy']*100:.2f}%",
                "Macro F1 (%)": f"{d['macro_f1']*100:.2f}%",
                "Weighted F1 (%)": f"{d['weighted_f1']*100:.2f}%",
                "Train Time (s)": f"{d['train_duration_sec']:.2f}s",
                "Latency/Sample": f"{d['latency_per_sample_ms']:.3f} ms"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # Plots Display
        st.markdown("---")
        st.markdown("#### Evaluation Visualizations")
        plots_dir = ROOT_DIR / "artifacts" / "plots"

        bar_plot = plots_dir / "model_comparison_bar.png"
        cm_plot = plots_dir / "confusion_matrices.png"
        lat_plot = plots_dir / "latency_and_efficiency.png"

        if bar_plot.exists():
            st.image(str(bar_plot), caption="Figure 1: Cross-Validation vs Test Performance Across Algorithms", use_container_width=True)

        if cm_plot.exists():
            st.image(str(cm_plot), caption="Figure 2: Normalized Confusion Matrices (Naive Bayes vs Logistic Regression vs Linear SVM)", use_container_width=True)

        if lat_plot.exists():
            st.image(str(lat_plot), caption="Figure 3: Training Duration & Inference Latency Trade-Off", use_container_width=True)


# ==============================================================================
# TAB 3: ACADEMIC DELIVERABLES (PDF & SLIDES)
# ==============================================================================
with tab_deliverables:
    st.markdown("### UE24CS352A Machine Learning Mini-Project Deliverables")
    st.write("In accordance with the assignment guidelines, the following deliverables are prepared:")

    col_del1, col_del2 = st.columns(2)

    with col_del1:
        st.markdown("""
        <div class="metric-card">
            <h4>📄 Two-Page Project Report (PDF)</h4>
            <p style="color:#cbd5e1; font-size:0.9rem;">
                A concise academic report formatted to two pages covering:
            </p>
            <ul style="color:#94a3b8; font-size:0.85rem;">
                <li>Problem Statement & Research Context</li>
                <li>Dataset Distribution & Preprocessing</li>
                <li>Methodology: Generative vs Discriminative Classifiers</li>
                <li>Empirical Results, Confusion Matrices & Latency</li>
                <li>Conclusions & Future Work</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        pdf_path = ROOT_DIR / "reports" / "Movie_Review_Sentiment_Analysis_Report.pdf"
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download Two-Page Report (PDF)",
                    data=f.read(),
                    file_name="UE24CS352A_Sentiment_Analysis_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("Run `python reports/generate_pdf_report.py` to compile the PDF report.")

    with col_del2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Presentation Slide Deck (PPTX)</h4>
            <p style="color:#cbd5e1; font-size:0.9rem;">
                A 5–6 slide presentation ready for the faculty review & viva session:
            </p>
            <ul style="color:#94a3b8; font-size:0.85rem;">
                <li>Slide 1: Title & Team Credentials</li>
                <li>Slide 2: Problem Formulation & Challenges</li>
                <li>Slide 3: Preprocessing & TF-IDF Extraction</li>
                <li>Slide 4: Model Architectures Compared</li>
                <li>Slide 5: Experimental Benchmarks & Insights</li>
                <li>Slide 6: Conclusion & Demonstration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        pptx_path = ROOT_DIR / "reports" / "Movie_Review_Sentiment_Analysis_Presentation.pptx"
        if pptx_path.exists():
            with open(pptx_path, "rb") as f:
                st.download_button(
                    "📥 Download Presentation Deck (PPTX)",
                    data=f.read(),
                    file_name="UE24CS352A_Sentiment_Analysis_Presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True
                )
        else:
            st.info("Run `python reports/generate_presentation.py` to compile the PPTX deck.")


# ==============================================================================
# TAB 4: DATASET & SETUP GUIDE
# ==============================================================================
with tab_dataset:
    st.markdown("### Dataset Instructions & Pipeline Architecture")
    st.markdown("""
    #### 1. Dataset Requirement
    Place your dataset inside `data/` or `data/raw/`:
    - **Rotten Tomatoes / SST (`train.tsv` / `train.csv`)**: Columns `PhraseId`, `SentenceId`, `Phrase`, `Sentiment` (0 to 4).
    - **General Movie Reviews (`movie_reviews.csv`)**: Columns `review`, `sentiment`.

    #### 2. Pipeline Execution Commands
    ```bash
    # 1. Activate Virtual Environment
    .\\.venv\\Scripts\\activate

    # 2. Run Comparative Training Pipeline
    python src/train.py

    # 3. Generate Deliverables (PDF & Presentation)
    python reports/generate_pdf_report.py
    python reports/generate_presentation.py

    # 4. Launch Streamlit Web UI
    streamlit run app/app.py
    ```
    """)
