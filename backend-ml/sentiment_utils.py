"""
sentiment_utils.py

Precomputes per-product VADER sentiment scores from Amazon review text.
Scores are cached to disk so they are only computed once.

Usage:
    from sentiment_utils import get_sentiment_scores
    scores = get_sentiment_scores(df)           # {asin: float in [-1, +1]}
    score = scores.get("B001234", 0.0)
"""

import os
import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CACHE_PATH = "data/sentiment_scores.json"


def compute_sentiment_scores(df: pd.DataFrame) -> dict:
    """
    Given a DataFrame with columns ['asin', 'reviewText'],
    compute the mean VADER compound score per product.

    Returns:
        dict mapping asin -> float (mean compound score, range [-1, +1])
    """
    analyzer = SentimentIntensityAnalyzer()

    # Drop rows with missing review text
    reviews = df[["asin", "reviewText"]].dropna(subset=["reviewText"]).copy()
    reviews["reviewText"] = reviews["reviewText"].astype(str)

    print(f"  Computing VADER scores for {len(reviews)} reviews across "
          f"{reviews['asin'].nunique()} products...")

    def score_text(text: str) -> float:
        return analyzer.polarity_scores(text)["compound"]

    reviews["compound"] = reviews["reviewText"].apply(score_text)

    # Aggregate: mean compound score per product
    asin_scores = (
        reviews.groupby("asin")["compound"]
        .mean()
        .to_dict()
    )

    print(f"  Done. Scored {len(asin_scores)} products.")
    return asin_scores


def get_sentiment_scores(df: pd.DataFrame, use_cache: bool = True) -> dict:
    """
    Load sentiment scores from cache if available, otherwise compute and save.

    Args:
        df:         Full review DataFrame (needs 'asin' and 'reviewText' columns).
        use_cache:  If True, read/write JSON cache at CACHE_PATH.

    Returns:
        dict mapping asin -> float sentiment score in [-1, +1]
    """
    if use_cache and os.path.exists(CACHE_PATH):
        print(f"  Loading cached sentiment scores from {CACHE_PATH}")
        with open(CACHE_PATH, "r") as f:
            return json.load(f)

    scores = compute_sentiment_scores(df)

    if use_cache:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(scores, f)
        print(f"  Saved sentiment cache to {CACHE_PATH}")

    return scores


def normalize_sentiment(scores: dict) -> dict:
    """
    Normalize raw VADER scores (already in [-1, +1]) to [0, 1] for
    uniform blending with normalized NCF scores.

    Formula: normalized = (score + 1) / 2
    """
    return {asin: (s + 1) / 2 for asin, s in scores.items()}
