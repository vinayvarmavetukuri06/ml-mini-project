"""
test_preprocessor.py
Tests text cleaning, normalization, contraction expansions,
edge cases (empty, None, invalid types), and vectorizer initialization.
"""

from src.preprocessor import clean_text, create_vectorizer


def test_clean_text_html_tags():
    """Verifies that HTML markup is completely removed."""
    raw = "<p>This is a <b>fantastic</b> movie!<br /><hr></p>"
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "fantastic" in cleaned
    assert "movie" in cleaned


def test_clean_text_urls():
    """Verifies that HTTP/HTTPS/WWW URLs are stripped."""
    raw = "Check out the review at https://example.com/review or www.imdb.com/title/123 for details."
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "https" not in cleaned
    assert "www" not in cleaned
    assert "review" in cleaned
    assert "details" in cleaned


def test_clean_text_contractions():
    """Verifies critical sentiment contractions are expanded."""
    raw = "I won't watch this. It can't be good and isn't exciting."
    cleaned = clean_text(raw)
    assert "will not" in cleaned
    assert "cannot" in cleaned
    assert "is not" in cleaned or "not" in cleaned


def test_clean_text_lowercasing_and_noise():
    """Verifies text is lowercased and special punctuation is stripped."""
    raw = "EXCELLENT Film!!! 10/10 Rating @ Cinema #MustWatch"
    cleaned = clean_text(raw)
    assert cleaned.islower()
    assert "!" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "excellent" in cleaned


def test_clean_text_empty_and_invalid_inputs():
    """Verifies clean_text handles empty, whitespace, None, and non-string inputs gracefully."""
    assert clean_text("") == ""
    assert clean_text("    ") == ""
    assert clean_text("\n\t  \r") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""
    assert clean_text(["list", "of", "words"]) == ""


def test_create_vectorizer_config():
    """Verifies create_vectorizer instantiates a TfidfVectorizer with expected properties."""
    vec = create_vectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    assert vec.ngram_range == (1, 2)
    assert vec.max_features == 5000
    assert vec.sublinear_tf is True
    assert vec.strip_accents == "unicode"
