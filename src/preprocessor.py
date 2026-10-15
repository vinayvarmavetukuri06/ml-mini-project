"""
preprocessor.py
Text cleaning, normalization, and TF-IDF vectorization module.
Follows ML Best Practices: separate fit on training split, transform on test split.
"""

import re
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


# Common contractions expansion mapping for sentiment-critical expressions
CONTRACTIONS = {
    r"won\'t": "will not",
    r"can\'t": "cannot",
    r"n\'t": " not",
    r"\'re": " are",
    r"\'s": " is",
    r"\'d": " would",
    r"\'ll": " will",
    r"\'t": " not",
    r"\'ve": " have",
    r"\'m": " am",
}


def clean_text(text: str) -> str:
    """
    Cleans raw review text:
    - Removes HTML tags (<br />, etc.)
    - Expands key negative contractions crucial for sentiment
    - Removes URLs
    - Normalizes non-alphanumeric noise while retaining word boundaries
    - Strips excess whitespace
    - Lowercases text
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Lowercase
    cleaned = cleaned.lower()

    # Contraction replacement
    for pattern, replacement in CONTRACTIONS.items():
        cleaned = re.sub(pattern, replacement, cleaned)

    # Remove non-word characters (keep alphanumeric, apostrophes for word stems, and basic spaces)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)

    # Normalize multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def create_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 15000,
    min_df: int = 2,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Creates and returns an unigram + bigram TF-IDF Vectorizer
    with sublinear TF scaling (sublinear_tf replaces tf with 1 + log(tf)).
    """
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode"
    )
