"""
evaluate_top10.py

Matrix Factorization recommender with sentiment-aware re-ranking.

Scoring formula:
    final_score = w1 * norm_ncf + w2 * diversity + w3 * novelty + alpha * sentiment_norm

Where:
    - norm_ncf        : MF predicted rating normalized to [0, 1]
    - diversity       : 1 - mean cosine similarity to user's past items
    - novelty         : 1 - log-normalized item popularity
    - sentiment_norm  : VADER compound score normalized to [0, 1]
    - alpha           : weight for sentiment (0 = pure NCF, 1 = pure sentiment)

For evaluation we sweep alpha over [0.0, 0.1, ..., 0.5] and report which
alpha yields the best NDCG@10 on the test set.
"""

from http import client

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import math
import json
import gzip
from sklearn.metrics.pairwise import cosine_similarity

# ── Sentiment ──────────────────────────────────────────────────────────────────
from sentiment_utils import get_sentiment_scores, normalize_sentiment

print("1. Loading Data and Model...")
train_df = pd.read_csv('data/train_data.csv')
test_df  = pd.read_csv('data/test_data.csv')

# ── Encoders ───────────────────────────────────────────────────────────────────
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

all_users = pd.concat([train_df['reviewerID'], test_df['reviewerID']])
all_items = pd.concat([train_df['asin'],       test_df['asin']])

user_encoder.fit(all_users)
item_encoder.fit(all_items)

user_id_to_idx = {u: i for i, u in enumerate(user_encoder.classes_)}

num_users   = len(user_encoder.classes_)
num_items   = len(item_encoder.classes_)
global_mean = train_df['overall'].mean()

# ── Model definition ───────────────────────────────────────────────────────────
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()
        self.user_emb  = nn.Embedding(num_users, emb_dim)
        self.item_emb  = nn.Embedding(num_items, emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user, item):
        u          = self.user_emb(user)
        i          = self.item_emb(item)
        dot_product = (u * i).sum(dim=1)
        return (dot_product
                + self.user_bias(user).squeeze()
                + self.item_bias(item).squeeze()
                + global_mean)

model = MatrixFactorization(num_users, num_items, emb_dim=32)
model.load_state_dict(torch.load('data/mf_model_weights_BEST_0.0939.pth'))
model.eval()

# ── Pre-computed caches ────────────────────────────────────────────────────────
train_user_items = train_df.groupby('reviewerID')['asin'].apply(set).to_dict()
test_user_items  = test_df.groupby('reviewerID')['asin'].apply(set).to_dict()
all_unique_items = set(train_df['asin']).union(set(test_df['asin']))

item_popularity  = train_df['asin'].value_counts().to_dict()
max_popularity   = max(item_popularity.values()) if item_popularity else 1

item_embeddings  = model.item_emb.weight.data.cpu().numpy()

# ── Sentiment scores (loaded once at module import) ────────────────────────────
print("1b. Loading sentiment scores...")

# We use the full review file so we have review text for every product.
# Adjust path if your raw data lives elsewhere.
_raw_df = pd.read_json('data/All_Beauty_5.json.gz', lines=True, compression='gzip')

_raw_sentiment_scores = get_sentiment_scores(_raw_df)          # {asin: [-1, +1]}
sentiment_scores_norm = normalize_sentiment(_raw_sentiment_scores)  # {asin: [0,  1]}

print(f"  Sentiment loaded for {len(sentiment_scores_norm)} products.")

# ── Metadata helpers ───────────────────────────────────────────────────────────
def load_metadata(file_path):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def build_lookup(meta_df):
    asin_to_title       = dict(zip(meta_df['asin'], meta_df['title']))
    asin_to_price       = dict(zip(meta_df['asin'], meta_df['price']))
    asin_to_image       = dict(zip(meta_df['asin'], meta_df['imageURLHighRes']))
    asin_to_details     = dict(zip(meta_df['asin'], meta_df['details']))
    asin_to_main_cat    = dict(zip(meta_df['asin'], meta_df['main_cat']))
    asin_to_brand       = dict(zip(meta_df['asin'], meta_df['brand']))
    return asin_to_title, asin_to_price, asin_to_image, asin_to_details, asin_to_main_cat, asin_to_brand


def prepare_metadata(meta_df):
    cols     = ['asin', 'title', 'price', 'imageURLHighRes', 'details', 'main_cat', 'brand']
    meta_df  = meta_df[[c for c in cols if c in meta_df.columns]]
    meta_df  = meta_df.drop_duplicates(subset='asin')
    meta_df['title']            = meta_df['title'].fillna("Unknown Product")
    meta_df['price']            = meta_df['price'].fillna("N/A")
    meta_df['imageURLHighRes']  = meta_df['imageURLHighRes'].fillna("N/A")
    meta_df['details']          = meta_df['details'].fillna("N/A")
    meta_df['main_cat']         = meta_df['main_cat'].fillna("N/A")
    meta_df['brand']            = meta_df['brand'].fillna("N/A")
    return meta_df


meta_df = load_metadata('./data/meta_All_Beauty.json.gz')
meta_df = prepare_metadata(meta_df)

(asin_to_title, asin_to_price, asin_to_image,
 asin_to_details, asin_to_main_category, asin_to_brand) = build_lookup(meta_df)


# ══════════════════════════════════════════════════════════════════════════════
# SHAP-style decomposition
# ══════════════════════════════════════════════════════════════════════════════
def explain_prediction(user_idx, item_idx):
    with torch.no_grad():
        u_vec      = model.user_emb.weight[user_idx]
        i_vec      = model.item_emb.weight[item_idx]
        user_bias  = model.user_bias.weight[user_idx].item()
        item_bias  = model.item_bias.weight[item_idx].item()
        interaction = torch.dot(u_vec, i_vec).item()
        pred       = global_mean + user_bias + item_bias + interaction
    return {
        "prediction": pred,
        "components": {
            "global_mean": global_mean,
            "user_bias":   user_bias,
            "item_bias":   item_bias,
            "interaction": interaction,
        }
    }


def get_similar_items(item_id, top_n=3):
    if item_id not in item_encoder.classes_:
        return []
    item_idx   = item_encoder.transform([item_id])[0]
    item_vec   = item_embeddings[item_idx].reshape(1, -1)
    sims       = cosine_similarity(item_vec, item_embeddings)[0]
    top_idx    = np.argsort(sims)[::-1][1:top_n + 1]
    return item_encoder.inverse_transform(top_idx)


# ══════════════════════════════════════════════════════════════════════════════
# Top-K Recommendation with sentiment re-ranking
#
# final_score = w1 * norm_ncf
#             + w2 * diversity
#             + w3 * novelty
#             + alpha * sentiment_norm
#
# alpha = 0  →  pure NCF  (baseline)
# alpha > 0  →  sentiment re-ranking (your contribution)
# ══════════════════════════════════════════════════════════════════════════════
def get_top_k_recommendations(
    user_id: str,
    k: int = 10,
    w1: float = 1.0,
    w2: float = 0.0,
    w3: float = 0.0,
    alpha: float = 0.0,   # ← NEW: sentiment weight
    return_scores: bool = True,
) -> dict:
    """
    Generate Top-K recommendations for a given user.

    Parameters
    ----------
    user_id : str
    k       : int    — list length
    w1      : float  — weight on NCF predicted rating
    w2      : float  — weight on diversity
    w3      : float  — weight on novelty
    alpha   : float  — weight on sentiment (0 = disabled, 0.3 is a good default)
    """
    if user_id not in user_id_to_idx:
        return {"recommendations": []}

    past_items   = train_user_items.get(user_id, set())
    unseen_items = list(all_unique_items - past_items)

    if not unseen_items:
        return {"recommendations": []}

    u_idx = torch.tensor([user_id_to_idx[user_id]] * len(unseen_items), dtype=torch.long)
    i_idx = torch.tensor(item_encoder.transform(unseen_items),           dtype=torch.long)

    with torch.no_grad():
        predictions = model(u_idx, i_idx).numpy()

    # Normalize NCF predictions to [0, 1]
    norm_predictions = (predictions - 1) / 4

    # ── Per-item score helpers ─────────────────────────────────────────────────
    def get_novelty(item_id):
        pop = item_popularity.get(item_id, 1)
        return 1 - (np.log(pop + 1) / np.log(max_popularity + 1))

    def get_diversity(item_id):
        if not past_items:
            return 0.0
        item_idx  = item_encoder.transform([item_id])[0]
        item_vec  = item_embeddings[item_idx].reshape(1, -1)
        past_idx  = item_encoder.transform(list(past_items))
        past_vecs = item_embeddings[past_idx]
        sims      = cosine_similarity(item_vec, past_vecs)[0]
        return 1 - float(np.mean(sims))

    # ── Compute final blended scores ───────────────────────────────────────────
    final_scores = []
    for idx, item_id in enumerate(unseen_items):
        ncf_score  = float(norm_predictions[idx])
        novelty    = get_novelty(item_id)
        diversity  = get_diversity(item_id)
        sentiment  = sentiment_scores_norm.get(item_id, 0.5)  # neutral default

        score = (w1 * ncf_score
                 + w2 * diversity
                 + w3 * novelty
                 + alpha * sentiment)
        final_scores.append(score)

    final_scores = np.array(final_scores)

    # ── Top-K selection ────────────────────────────────────────────────────────
    top_k_idx = np.argpartition(final_scores, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(final_scores[top_k_idx])[::-1]]

    results = []
    for i in top_k_idx:
        item_id = unseen_items[i]

        # ── Cosine-based explanation ───────────────────────────────────────────
        similar_explanations = []
        if past_items:
            item_idx_current = item_encoder.transform([item_id])[0]
            item_vec_current = item_embeddings[item_idx_current].reshape(1, -1)
            past_indices     = item_encoder.transform(list(past_items))
            past_vecs        = item_embeddings[past_indices]
            sims             = cosine_similarity(item_vec_current, past_vecs)[0]
            top_sim_idx      = np.argsort(sims)[::-1][:2]
            similar_items    = [list(past_items)[j] for j in top_sim_idx]
            similar_explanations = [asin_to_title.get(x, x) for x in similar_items]

        explanation = {
            "because_you_liked": similar_explanations,
            "similar_items": [asin_to_title.get(x, x) for x in get_similar_items(item_id, 2)],
            "predicted_rating": float(predictions[i]),
        }

        u_idx_single = user_id_to_idx[user_id]
        i_idx_single = item_encoder.transform([item_id])[0]
        shap_info    = explain_prediction(u_idx_single, i_idx_single)

        raw_sentiment = _raw_sentiment_scores.get(item_id, 0.0)   # for display

        item_data = {
            "item_id":          item_id,
            "name":             asin_to_title.get(item_id, "Unknown Product"),
            "price":            asin_to_price.get(item_id, "N/A"),
            "imageURLHighRes":  asin_to_image.get(item_id, "N/A"),
            "details":          asin_to_details.get(item_id, "N/A"),
            "main_cat":         asin_to_main_category.get(item_id, "N/A"),
            "brand":            asin_to_brand.get(item_id, "N/A"),
            "explanation":      explanation,
            "shap_explanation": shap_info,
            # ── sentiment fields ──────────────────────────────────────────────
            "sentiment_score":  round(raw_sentiment, 4),   # [-1, +1] for UI
            "sentiment_label":  _sentiment_label(raw_sentiment),
            # ── score breakdown ───────────────────────────────────────────────
            "score_breakdown": {
                "predicted_rating":   float(predictions[i]),
                "normalized_rating":  float(norm_predictions[i]),
                "diversity":          float(get_diversity(item_id)),
                "novelty":            float(get_novelty(item_id)),
                "sentiment_norm":     float(sentiment_scores_norm.get(item_id, 0.5)),
                "alpha":              alpha,
                "final_score":        float(final_scores[i]),
            },
        }

        if return_scores:
            item_data["score"] = float(predictions[i])

        results.append(item_data)

    return {"recommendations": results}


def _sentiment_label(score: float) -> str:
    """Map raw VADER compound score to a human-readable label."""
    if score >= 0.5:
        return "Very Positive"
    elif score >= 0.1:
        return "Positive"
    elif score >= -0.1:
        return "Neutral"
    elif score >= -0.5:
        return "Negative"
    else:
        return "Very Negative"


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION — baseline (alpha=0) vs best alpha
# ══════════════════════════════════════════════════════════════════════════════
print("2. Evaluating recommendations (baseline + sentiment sweep)...")


def _run_eval(alpha: float = 0.0) -> dict:
    """Run Top-10 evaluation for a given alpha and return metric averages."""
    precisions, recalls, f_measures, ndcgs = [], [], [], []

    with torch.no_grad():
        for user_id, true_test_items in test_user_items.items():
            if not true_test_items:
                continue

            recs = get_top_k_recommendations(user_id, k=10, alpha=alpha)
            top_10 = [x["item_id"] for x in recs["recommendations"]]

            if not top_10:
                continue

            hits      = len(set(top_10) & true_test_items)
            precision = hits / 10.0
            recall    = hits / len(true_test_items)
            f_measure = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            # NDCG
            dcg  = sum(1.0 / math.log2(rank + 2)
                       for rank, item in enumerate(top_10)
                       if item in true_test_items)
            idcg = sum(1.0 / math.log2(i + 2)
                       for i in range(min(10, len(true_test_items))))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f_measures.append(f_measure)
            ndcgs.append(ndcg)

    return {
        "alpha":            alpha,
        "users_evaluated":  len(precisions),
        "precision":        float(np.mean(precisions)),
        "recall":           float(np.mean(recalls)),
        "f_measure":        float(np.mean(f_measures)),
        "ndcg":             float(np.mean(ndcgs)),
    }


# ── Baseline (no sentiment) ────────────────────────────────────────────────────
_baseline = _run_eval(alpha=0.0)
print(f"\n--- Baseline (alpha=0.0, pure NCF) ---")
print(f"  Users evaluated : {_baseline['users_evaluated']}")
print(f"  Precision@10    : {_baseline['precision']:.4f}")
print(f"  Recall@10       : {_baseline['recall']:.4f}")
print(f"  F-measure       : {_baseline['f_measure']:.4f}")
print(f"  NDCG@10         : {_baseline['ndcg']:.4f}")

# ── Alpha sweep ────────────────────────────────────────────────────────────────
_ALPHA_CANDIDATES = [0.1, 0.2, 0.3, 0.4, 0.5]
_sweep_results    = []

print("\n--- Sentiment alpha sweep ---")
for _a in _ALPHA_CANDIDATES:
    _res = _run_eval(alpha=_a)
    _sweep_results.append(_res)
    print(f"  alpha={_a:.1f}  NDCG={_res['ndcg']:.4f}  "
          f"Prec={_res['precision']:.4f}  Rec={_res['recall']:.4f}")

# ── Best alpha ─────────────────────────────────────────────────────────────────
_best = max(_sweep_results, key=lambda x: x["ndcg"])
print(f"\n  ✅  Best alpha = {_best['alpha']} → NDCG@10 = {_best['ndcg']:.4f}")
print("-" * 45)

# Store final metrics for the API to expose
_final_metrics = {
    "baseline":     _baseline,
    "best_alpha":   _best,
    "alpha_sweep":  _sweep_results,
}


def evaluate_model() -> dict:
    """Return evaluation results (called by main.py on startup)."""
    return _final_metrics