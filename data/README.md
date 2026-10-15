# Dataset Specification & Instructions

## Required Dataset Files

The ML pipeline is designed to work seamlessly with the standard Movie Review Sentiment Analysis datasets. Place your dataset file directly in this `data/` directory or `data/raw/`.

### Supported Format 1: Kaggle Rotten Tomatoes / Stanford Sentiment Treebank (Primary)
This matches the reference research paper in the workspace (*"Sentiment Analysis on Movie Reviews"* by Xiao Cai and Ya Wang):

- **Expected Filename**: `train.tsv` (or `train.csv`)
- **Location**: `data/train.tsv` or `data/train.csv`
- **Columns**:
  - `PhraseId`: Unique integer identifier for each phrase.
  - `SentenceId`: Integer identifier for the full sentence.
  - `Phrase`: The text content of the review/phrase.
  - `Sentiment`: Sentiment score from `0` to `4`:
    - `0` = Negative
    - `1` = Somewhat Negative
    - `2` = Neutral
    - `3` = Somewhat Positive
    - `4` = Positive

### Supported Format 2: Standard Binary / 3-Class Movie Reviews (e.g. IMDB / Rotten Tomatoes)
If you have a binary or 3-class review dataset:

- **Expected Filename**: `movie_reviews.csv` or `IMDB Dataset.csv`
- **Location**: `data/movie_reviews.csv`
- **Columns**:
  - `review` (or `text` / `Phrase`): Text of the review.
  - `sentiment` (or `label` / `Sentiment`): Categorical (`positive`, `negative`, `neutral`) or numeric (`0`, `1`).

---

## How to Acquire the Dataset

1. **Option A (Kaggle)**:
   - Download `train.tsv.zip` from [Kaggle Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data).
   - Unzip and copy `train.tsv` into `data/train.tsv`.

2. **Option B (Automated Fetcher)**:
   - Run the provided fetch script to automatically download and prepare the real Rotten Tomatoes dataset from official repositories:
     ```powershell
     .\.venv\Scripts\python src/data_fetcher.py
     ```

Once the dataset is placed, run:
```powershell
.\.venv\Scripts\python src/train.py
```
