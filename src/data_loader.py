"""
data_loader.py
Dataset ingestion and verification module for Movie Review Sentiment Analysis.
Supports Rotten Tomatoes / Stanford Sentiment Treebank (train.tsv / train.csv)
and standard movie review CSV formats.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


LABEL_MAP_5_CLASS = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}


def find_dataset_file(search_dirs: Optional[list] = None) -> Optional[Path]:
    """
    Search for supported movie review dataset files across target directories.
    """
    if search_dirs is None:
        workspace_dir = Path(__file__).resolve().parent.parent
        search_dirs = [
            workspace_dir / "data",
            workspace_dir / "data" / "raw",
            workspace_dir
        ]

    candidate_names = [
        "train.tsv",
        "train.csv",
        "movie_reviews.csv",
        "imdb.csv",
        "IMDB Dataset.csv",
        "sentiment_analysis.csv",
        "rotten_tomatoes.csv"
    ]

    for directory in search_dirs:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        for name in candidate_names:
            file_path = dir_path / name
            if file_path.is_file() and file_path.stat().st_size > 0:
                return file_path
    return None


def load_dataset(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Loads and standardizes movie reviews into a DataFrame with columns:
    ['text', 'label', 'label_name'].

    Returns:
        df: pd.DataFrame with standardized columns
        metadata: dict with summary stats
    """
    resolved_path: Optional[Path] = None
    if file_path:
        p = Path(file_path)
        if p.is_file():
            resolved_path = p
    if resolved_path is None:
        resolved_path = find_dataset_file()

    if resolved_path is None:
        workspace_dir = Path(__file__).resolve().parent.parent
        raise FileNotFoundError(
            f"No dataset file found! Please provide a dataset file (such as 'train.tsv' or 'movie_reviews.csv') "
            f"inside '{workspace_dir / 'data'}'. Refer to 'data/README.md' for instructions."
        )

    print(f"[DataLoader] Loading dataset from: {resolved_path}")
    ext = resolved_path.suffix.lower()
    sep = "\t" if ext == ".tsv" else ","

    try:
        df = pd.read_csv(resolved_path, sep=sep, on_bad_lines="skip", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(resolved_path, sep=sep, on_bad_lines="skip", encoding="latin-1")

    # Detect column mappings
    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    # Text column resolution
    text_col = None
    for candidate in ["phrase", "review", "text", "content", "sentence"]:
        if candidate in cols_lower:
            text_col = cols_lower[candidate]
            break

    # Sentiment / Label column resolution
    label_col = None
    for candidate in ["sentiment", "label", "target", "score", "class"]:
        if candidate in cols_lower:
            label_col = cols_lower[candidate]
            break

    if text_col is None or label_col is None:
        raise ValueError(
            f"Dataset at {resolved_path} lacks recognized text and label columns. "
            f"Found columns: {list(df.columns)}. Expected columns like 'Phrase'/'review' and 'Sentiment'/'label'."
        )

    # Standardize DataFrame
    clean_df = pd.DataFrame()
    clean_df["text"] = df[text_col].astype(str)
    clean_df["label"] = df[label_col]

    # Handle missing/empty text
    initial_count = len(clean_df)
    clean_df = clean_df.dropna(subset=["text", "label"])
    clean_df = clean_df[clean_df["text"].str.strip() != ""]
    clean_df = clean_df[clean_df["text"].str.lower() != "nan"]

    # Normalize labels
    if pd.api.types.is_numeric_dtype(clean_df["label"]):
        clean_df["label"] = clean_df["label"].astype(int)
        unique_labels = sorted(clean_df["label"].unique())
        if set(unique_labels).issubset({0, 1, 2, 3, 4}):
            clean_df["label_name"] = clean_df["label"].map(LABEL_MAP_5_CLASS)
        elif set(unique_labels).issubset({0, 1}):
            clean_df["label_name"] = clean_df["label"].map({0: "Negative", 1: "Positive"})
        else:
            clean_df["label_name"] = clean_df["label"].astype(str)
    else:
        # String categorical labels
        clean_df["label_name"] = clean_df["label"].astype(str).str.strip().str.title()
        label_categories = sorted(clean_df["label_name"].unique())
        label_to_id = {name: idx for idx, name in enumerate(label_categories)}
        clean_df["label"] = clean_df["label_name"].map(label_to_id)

    metadata = {
        "file_path": str(resolved_path),
        "initial_rows": initial_count,
        "valid_rows": len(clean_df),
        "dropped_rows": initial_count - len(clean_df),
        "num_classes": clean_df["label"].nunique(),
        "classes": [clean_df.loc[clean_df["label"] == i, "label_name"].iloc[0] for i in sorted(clean_df["label"].unique())],
        "class_distribution": {
            clean_df.loc[clean_df["label"] == i, "label_name"].iloc[0]: int((clean_df["label"] == i).sum())
            for i in sorted(clean_df["label"].unique())
        }
    }

    print(f"[DataLoader] Successfully loaded {len(clean_df)} valid reviews across {metadata['num_classes']} classes.")
    return clean_df, metadata


if __name__ == "__main__":
    try:
        data, meta = load_dataset()
        print("Dataset loaded successfully:")
        print(meta)
    except FileNotFoundError as e:
        print(e)
