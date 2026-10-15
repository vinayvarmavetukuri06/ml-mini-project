"""
test_main_cli.py
Automated tests for main.py interactive terminal CLI and helper functions:
1. Progress bar formatting
2. Evaluation summary rendering
3. Direct champion model prediction flow without redundant model selection prompts
4. Graceful CLI exit and invalid option handling
"""

import subprocess
import sys
from pathlib import Path
from main import format_bar, print_evaluation_summary, BANNER, MENU


def test_format_bar_values():
    """Tests the format_bar helper produces expected ASCII output and formatting."""
    bar_100 = format_bar(100.0, width=20)
    assert "#" * 20 in bar_100
    assert "100.0%" in bar_100

    bar_0 = format_bar(0.0, width=20)
    assert "-" * 20 in bar_0
    assert "0.0%" in bar_0

    bar_50 = format_bar(50.0, width=20)
    assert "#" * 10 in bar_50
    assert "-" * 10 in bar_50
    assert "50.0%" in bar_50


def test_banner_and_menu_constants():
    """Verifies that terminal banner and menu contain required 5-class and menu options."""
    assert "MOVIE REVIEW SENTIMENT ANALYSIS" in BANNER
    assert "5 Classes" in BANNER
    assert "[1] Train and compare ML models" in MENU
    assert "[2] Predict sentiment for a custom movie review" in MENU
    assert "[3] Evaluate saved models" in MENU
    assert "[4] Exit" in MENU


def test_print_evaluation_summary_output(metrics_data, capsys):
    """Verifies print_evaluation_summary renders table and identifies champion model."""
    print_evaluation_summary(metrics_data)
    captured = capsys.readouterr().out
    assert "MODEL" in captured
    assert "TEST ACC" in captured
    assert "MACRO F1" in captured
    assert "[CHAMPION] Best Model Selected: Linear SVM" in captured


def test_direct_champion_prediction_cli(project_root):
    """
    Subprocess test verifying direct prediction workflow:
    - Option 2 directly asks for review input (no model selection question).
    - Automatically loads Linear SVM champion model.
    - Accurately predicts sentiment and outputs 5-class distribution.
    - 'back' returns to main menu, Option 4 exits with code 0.
    """
    inputs = "2\nAn extraordinary and breathtaking masterpiece.\nback\n4\n"
    res = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(project_root),
        input=inputs,
        text=True,
        capture_output=True,
        timeout=15
    )

    assert res.returncode == 0, f"Process exited with non-zero code {res.returncode}. Stderr: {res.stderr}"
    assert "Select Classification Model" not in res.stdout, "Error: CLI still prompts user to select a model!"
    assert "Enter your movie review (or type 'back' to return):" in res.stdout, "Error: Review prompt missing!"
    assert "Linear SVM" in res.stdout, "Error: Champion model Linear SVM not loaded!"
    assert "Probability Distribution" in res.stdout, "Error: Probability distribution missing in CLI output!"
    assert ("VERY POSITIVE" in res.stdout or "POSITIVE" in res.stdout), "Expected positive sentiment in CLI output!"


def test_cli_exit_option(project_root):
    """Verifies Option 4 cleanly exits application with status 0."""
    res = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(project_root),
        input="4\n",
        text=True,
        capture_output=True,
        timeout=10
    )
    assert res.returncode == 0
    assert "Exiting Movie Review Sentiment Analysis System. Goodbye!" in res.stdout


def test_cli_invalid_option_handling(project_root):
    """Verifies CLI gracefully handles invalid choices without crashing."""
    res = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(project_root),
        input="9\n4\n",
        text=True,
        capture_output=True,
        timeout=10
    )
    assert res.returncode == 0
    assert "[Invalid Option] Please enter 1, 2, 3, or 4." in res.stdout
