"""
test_main_cli.py
Automated CLI runner to verify the updated main.py direct prediction flow:
1. Select Option 2 (Predict sentiment) -> immediately prompts for review (no model selection question).
2. Enter custom review.
3. Verify predicted class, sentiment label, probability distribution.
4. Type 'back' to return to main menu.
5. Select Option 4 (Exit).
"""

import subprocess
import sys

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_main_cli_prediction():
    """Automated CLI runner to verify the updated main.py direct prediction flow."""
    inputs = "2\nAn extraordinary and breathtaking masterpiece.\nback\n4\n"

    res = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(ROOT_DIR),
        input=inputs,
        text=True,
        capture_output=True,
        timeout=15
    )

    assert res.returncode == 0, f"Process exited with non-zero code {res.returncode}"
    assert "Select Classification Model" not in res.stdout, "Error: Still asking user to select model!"
    assert "Enter your movie review (or type 'back' to return):" in res.stdout, "Error: Prompt missing!"
    assert "Linear SVM" in res.stdout, "Error: Champion model Linear SVM not loaded!"
    assert ("VERY POSITIVE" in res.stdout or "POSITIVE" in res.stdout), "Error: Expected sentiment not found!"
    assert "Probability Distribution" in res.stdout, "Error: Probability distribution missing!"


if __name__ == "__main__":
    test_main_cli_prediction()
    print("\n[VERIFICATION PASSED] Direct champion model prediction workflow functions perfectly!")
