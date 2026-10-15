"""
data_fetcher.py
Utility to download official benchmark Movie Review Sentiment datasets
directly from authentic public academic repositories (Pang & Lee Cornell / Rotten Tomatoes).
Does NOT fabricate any data; fetches authentic benchmark records with progress reporting.
"""

import sys
import urllib.request
import tarfile
from pathlib import Path
import pandas as pd


PANG_LEE_POLARITY_URL = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"


def download_official_cornell_dataset(dest_dir: Path) -> Path:
    """
    Downloads the official Bo Pang and Lillian Lee Rotten Tomatoes sentence polarity dataset
    (5,331 positive and 5,331 negative movie review snippets).
    Saves to data/movie_reviews.csv.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "rt-polaritydata.tar.gz"
    out_csv = dest_dir / "movie_reviews.csv"

    print("[DataFetcher] Downloading official Pang & Lee Rotten Tomatoes dataset from Cornell...")
    urllib.request.urlretrieve(PANG_LEE_POLARITY_URL, archive_path)
    print(f"[DataFetcher] Download complete ({archive_path.stat().st_size} bytes). Extracting...")

    pos_reviews = []
    neg_reviews = []

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("rt-polarity.pos"):
                f = tar.extractfile(member)
                if f:
                    for line in f:
                        decoded = line.decode("latin-1").strip()
                        if decoded:
                            pos_reviews.append(decoded)
            elif member.name.endswith("rt-polarity.neg"):
                f = tar.extractfile(member)
                if f:
                    for line in f:
                        decoded = line.decode("latin-1").strip()
                        if decoded:
                            neg_reviews.append(decoded)

    print(f"[DataFetcher] Extracted {len(pos_reviews)} positive reviews and {len(neg_reviews)} negative reviews.")
    
    rows = []
    for r in pos_reviews:
        rows.append({"review": r, "sentiment": "positive", "label": 1})
    for r in neg_reviews:
        rows.append({"review": r, "sentiment": "negative", "label": 0})

    df = pd.DataFrame(rows)
    # Shuffle predictably
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[DataFetcher] Saved benchmark dataset to: {out_csv} ({len(df)} rows)")

    # Clean up archive
    if archive_path.exists():
        archive_path.unlink()

    return out_csv


if __name__ == "__main__":
    target_data_dir = Path(__file__).resolve().parent.parent / "data"
    try:
        csv_file = download_official_cornell_dataset(target_data_dir)
        print(f"[DataFetcher] Successfully ready: {csv_file}")
    except Exception as exc:
        print(f"[DataFetcher] Error downloading dataset: {exc}")
        print("Please manually copy your 'train.tsv' or 'movie_reviews.csv' into 'data/'.")
        sys.exit(1)
