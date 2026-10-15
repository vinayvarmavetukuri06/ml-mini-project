# Movie Review Sentiment Analysis — ML Mini-Project

[![Course](https://img.shields.io/badge/Course-UE24CS352A%20Machine%20Learning-blue.svg)](https://pes.edu)
[![Python](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://python.org)
[![Interface](https://img.shields.io/badge/Interface-Interactive%20Terminal-orange.svg)](#)
[![Classes](https://img.shields.io/badge/Classification-5--Class%20Multiclass-purple.svg)](#)

An end-to-end Machine Learning pipeline and interactive terminal application comparing **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear Support Vector Machines (Linear SVM)** using **TF-IDF n-gram feature representations** for fine-grained 5-class movie review sentiment classification.

Developed in compliance with the **UE24CS352A Machine Learning Mini-Project Guidelines** and benchmarked on the Rotten Tomatoes / Stanford Sentiment Treebank dataset (matching reference research by Xiao Cai & Ya Wang, Stanford CS229).

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [5-Class Sentiment Taxonomy](#-5-class-sentiment-taxonomy)
3. [Empirical Benchmark Results](#-empirical-benchmark-results)
4. [Deliverables Alignment](#-deliverables-alignment)
5. [Architecture & Methodology](#-architecture--methodology)
6. [Quickstart: Terminal Commands](#-quickstart-terminal-commands)
7. [Repository Structure](#-repository-structure)

---

## 🎯 Project Overview

This project implements a rigorous comparative evaluation of three canonical NLP classifiers trained on the full **Rotten Tomatoes / Stanford Sentiment Treebank corpus (156,059 review phrases)**:
- **Multinomial Naive Bayes (MNB)**: Generative probabilistic model with Laplace smoothing ($\alpha=1.0$).
- **Logistic Regression**: Discriminative linear model with L2 regularization and multiclass softmax.
- **Linear Support Vector Machine (Linear SVM)**: Max-margin hyperplane classifier calibrated via Platt scaling cross-validation for genuine class probabilities.

All models utilize **sublinear TF-IDF features (unigrams + bigrams)** to capture negation and compound sentiment phrases (e.g., *"not good"*, *"heart wrenching"*).

---

## 🏷️ 5-Class Sentiment Taxonomy

In strict accordance with the Kaggle / Stanford Sentiment Treebank reference specification:

| Label ID | Sentiment Class | Description |
| :---: | :--- | :--- |
| `0` | **Very Negative** | Strongly critical, scathing reviews or phrases |
| `1` | **Negative** | Mildly negative or unfavorable comments |
| `2` | **Neutral** | Objective, descriptive, or emotionally balanced phrases |
| `3` | **Positive** | Favorable, approving comments |
| `4` | **Very Positive** | Highly enthusiastic, rave reviews or praise |

**Dataset Breakdown** (`data/train.tsv`):
- Total records: **156,059 valid reviews/phrases**
- Distribution: Neutral (79,582), Positive (32,927), Negative (27,272), Very Positive (9,206), Very Negative (7,072).

---

## 📊 Empirical Benchmark Results

Evaluated on **124,847 training instances** and **31,212 unseen test instances** with 15,000 TF-IDF features:

| Model Architecture | 5-Fold Cross-Validation Accuracy | Test Holdout Accuracy | Macro F1-Score | Weighted F1-Score | Train Duration | Inference Latency / Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** | 63.09% (±0.21%) | **63.85%** | **50.27%** | **61.42%** | 20.00s | 0.002 ms |
| **Logistic Regression** | **63.55%** (±0.28%) | **63.96%** | 48.24% | 61.42% | 11.14s | 0.000 ms |
| **Multinomial Naive Bayes** | 59.71% (±0.40%) | 60.06% | 36.49% | 54.83% | **0.04s** | 0.000 ms |

> 🏆 **Champion Model**: **Linear SVM** achieving the highest Macro F1-score (**50.27%**), accurately balancing minority polar classes (Very Negative and Very Positive) against the dominant Neutral class.

---

## 🏆 Deliverables Alignment (UE24CS352A Rubric)

| Deliverable | Description | Location |
| :--- | :--- | :--- |
| **A. GitHub Source Code** | Modular, PEP8-compliant Python source code & documentation | Entire repository & `README.md` |
| **B. Two-Page PDF Write-up** | Structured academic summary covering problem, data, methodology, results & conclusions | [`reports/Movie_Review_Sentiment_Analysis_Report.pdf`](reports/Movie_Review_Sentiment_Analysis_Report.pdf) |
| **C. Slide Presentation (6 Slides)** | High-impact 16:9 widescreen presentation deck for viva & review | [`reports/Movie_Review_Sentiment_Analysis_Presentation.pptx`](reports/Movie_Review_Sentiment_Analysis_Presentation.pptx) |
| **D. Live Terminal Demonstration** | Fully interactive terminal interface with model selection, live text prediction, probability distributions & keyword explainability | [`main.py`](main.py) |

---

## 🚀 Quickstart: Terminal Commands

### 1. Launch the Interactive Terminal Application
To run the project demonstration entirely in the terminal:
```powershell
.\.venv\Scripts\python main.py
```
```text
MAIN MENU:
  [1] Train and compare ML models
  [2] Predict sentiment for a custom movie review
  [3] Evaluate saved models and generate deliverables (PDF & PPTX)
  [4] Exit
```
* **Option 2 (Direct Prediction)**: Automatically loads the champion model (Linear SVM based on highest Macro F1: 50.27%) and directly asks `Enter your movie review (or type 'back' to return):`. Displays the predicted class, confidence, calibrated 5-class probability distribution, and top TF-IDF token contributions.
* **Option 1 & 3 (Model Benchmarking)**: Trains and compares all three algorithms (Multinomial NB, Logistic Regression, Linear SVM) with 5-fold cross-validation and exports evaluation reports.

### 2. Standalone Training Command
To train the models directly from the command line:
```powershell
.\.venv\Scripts\python src/train.py
```

### 3. Standalone Deliverables Compilation
To regenerate the 2-page academic PDF report and 6-slide PowerPoint deck:
```powershell
.\.venv\Scripts\python reports/generate_pdf_report.py
.\.venv\Scripts\python reports/generate_presentation.py
```

---

## 📁 Repository Structure

```
├── main.py                        # Root interactive terminal interface
├── data/
│   ├── README.md                  # Dataset specifications and format details
│   └── train.tsv                  # Authentic 5-class Rotten Tomatoes SST dataset (156k rows)
├── artifacts/
│   ├── models/                    # Saved models (.joblib), vectorizer, label mappings
│   ├── plots/                     # 5x5 confusion matrices, comparison bars, latency plots
│   └── metrics.json               # Genuine 5-class empirical evaluation metrics
├── reports/
│   ├── generate_pdf_report.py     # Two-page academic PDF report generator
│   ├── generate_presentation.py   # 6-slide PowerPoint presentation generator
│   ├── Movie_Review_Sentiment_Analysis_Report.pdf          # 2-Page PDF write-up
│   └── Movie_Review_Sentiment_Analysis_Presentation.pptx   # 6-Slide presentation deck
├── src/
│   ├── __init__.py                # Package initializer
│   ├── data_loader.py             # 5-class dataset loader & validation
│   ├── preprocessor.py            # Text cleaning & TF-IDF feature extraction
│   ├── train.py                   # Comparative training & 5-fold CV pipeline
│   ├── evaluate.py                # Visual reporting & 5x5 confusion matrix plots
│   └── predict.py                 # 5-class inference & calibrated probability engine
├── requirements.txt               # Pinned project dependencies
└── README.md                      # Comprehensive project documentation
```
