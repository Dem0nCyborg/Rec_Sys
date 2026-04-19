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


print("1. Loading Data and Model...")
train_df = pd.read_csv('data/train_data.csv')
test_df = pd.read_csv('data/test_data.csv')

# Encoders
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

all_users = pd.concat([train_df['reviewerID'], test_df['reviewerID']])
all_items = pd.concat([train_df['asin'], test_df['asin']])


user_encoder.fit(all_users)
item_encoder.fit(all_items)

# Fast lookup mapping (optimization)
user_id_to_idx = {u: i for i, u in enumerate(user_encoder.classes_)}

num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
global_mean = train_df['overall'].mean()



# Model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        dot_product = (u * i).sum(dim=1)
        return dot_product + self.user_bias(user).squeeze() + self.item_bias(item).squeeze() + global_mean

# Load trained model
model = MatrixFactorization(num_users, num_items, emb_dim=32)
model.load_state_dict(torch.load('data/mf_model_weights_BEST_0.0939.pth'))
model.eval()

# Precompute
train_user_items = train_df.groupby('reviewerID')['asin'].apply(set).to_dict()
test_user_items = test_df.groupby('reviewerID')['asin'].apply(set).to_dict()
all_unique_items = set(train_df['asin']).union(set(test_df['asin']))
# ----------- Popularity (for novelty/fairness) -----------
item_popularity = train_df['asin'].value_counts().to_dict()
max_popularity = max(item_popularity.values()) if item_popularity else 1
# Precompute item embeddings 
item_embeddings = model.item_emb.weight.data.cpu().numpy()

# Loading the metadata file for product information (title, price)
def load_metadata(file_path):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def build_lookup(meta_df):
    asin_to_title = dict(zip(meta_df['asin'], meta_df['title']))
    asin_to_price = dict(zip(meta_df['asin'], meta_df['price']))
    asin_to_image = dict(zip(meta_df['asin'], meta_df['imageURLHighRes']))
    asin_to_details = dict(zip(meta_df['asin'], meta_df['details']))
    asin_to_main_category = dict(zip(meta_df['asin'], meta_df['main_cat']))
    asin_to_brand = dict(zip(meta_df['asin'], meta_df['brand']))
    return asin_to_title, asin_to_price, asin_to_image, asin_to_details, asin_to_main_category, asin_to_brand

def prepare_metadata(meta_df):
    # Keep only useful columns (safe check if columns exist)
    cols = ['asin', 'title', 'price', 'imageURLHighRes', "details", "main_cat", "brand"]
    meta_df = meta_df[[c for c in cols if c in meta_df.columns]]

    # Drop duplicates
    meta_df = meta_df.drop_duplicates(subset='asin')

    # Fill missing values
    meta_df['title'] = meta_df['title'].fillna("Unknown Product")
    meta_df['price'] = meta_df['price'].fillna("N/A")
    meta_df['imageURLHighRes'] = meta_df['imageURLHighRes'].fillna("N/A")
    meta_df['details'] = meta_df['details'].fillna("N/A")
    meta_df['main_cat'] = meta_df['main_cat'].fillna("N/A")
    meta_df['brand'] = meta_df['brand'].fillna("N/A")

    return meta_df

meta_df = load_metadata('./data/meta_All_Beauty.json.gz')  # change file if needed
meta_df = prepare_metadata(meta_df)

asin_to_title, asin_to_price, asin_to_image, asin_to_details, asin_to_main_category, asin_to_brand = build_lookup(meta_df)

# =========================================================
# ✅ FUNCTION: Top-K Recommendation (FOR API / REUSE)
# “We extended the base matrix factorization model with a controllable 
# ranking function that incorporates predicted relevance, diversity, and 
# novelty using a weighted scoring framework.”

# final_score = w1 * predicted_rating + w2 * diversity + w3 * novelty
# w1 (Accuracy): Prioritizes high predicted ratings by directly weighting the MF prediction score.
# w2 (Diversity): Promotes dissimilar items by weighting the inverse similarity (1 − cosine similarity) to the user’s past items.
# w3 (Novelty): Favors less popular items by weighting the inverse normalized popularity (1 − popularity score).

# =========================================================
def get_top_k_recommendations(user_id, k=10, w1=1.0, w2=0.0, w3=0.0, return_scores=True):
    if user_id not in user_id_to_idx:
        return []

    past_items = train_user_items.get(user_id, set())
    unseen_items = list(all_unique_items - past_items)

    if len(unseen_items) == 0:
        return []

    u_idx = torch.tensor(
        [user_id_to_idx[user_id]] * len(unseen_items),
        dtype=torch.long
    )
    i_idx = torch.tensor(
        item_encoder.transform(unseen_items),
        dtype=torch.long
    )

    with torch.no_grad():
        predictions = model(u_idx, i_idx).numpy()

    # Normalize predictions (important for mixing scores)
    norm_predictions = (predictions - 1) / 4

    # ----------- Helper functions -----------
    def get_novelty(item_id):
        pop = item_popularity.get(item_id, 1)
        return 1 - (np.log(pop + 1) / np.log(max_popularity + 1))

    def get_diversity(item_id):
        if len(past_items) == 0:
            return 0

        item_idx = item_encoder.transform([item_id])[0]
        item_vec = item_embeddings[item_idx].reshape(1, -1)

        past_indices = item_encoder.transform(list(past_items))
        past_vecs = item_embeddings[past_indices]

        sims = cosine_similarity(item_vec, past_vecs)[0]
        return 1 - np.mean(sims)

    # ----------- Compute final scores -----------
    final_scores = []

    for idx, item_id in enumerate(unseen_items):
        pred = norm_predictions[idx]
        novelty = get_novelty(item_id)
        diversity = get_diversity(item_id)

        score = w1 * pred + w2 * diversity + w3 * novelty
        final_scores.append(score)

    final_scores = np.array(final_scores)

    # ----------- Top-K selection -----------
    top_k_idx = np.argpartition(final_scores, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(final_scores[top_k_idx])[::-1]]

    results = []

    for i in top_k_idx:
        item_id = unseen_items[i]

        item_data = {
            "item_id": item_id,
            "name": asin_to_title.get(item_id, "Unknown Product"),
            "price": asin_to_price.get(item_id, "N/A"),
            "imageURLHighRes": asin_to_image.get(item_id, "N/A"),
            "details": asin_to_details.get(item_id, "N/A"),
            "main_cat": asin_to_main_category.get(item_id, "N/A"),
            "brand": asin_to_brand.get(item_id, "N/A")
        }

        # ----------- Score -----------
        if return_scores:
            item_data["score"] = float(predictions[i])

        # ----------- Explainability -----------
        user_past_items = list(past_items)
        similar_explanations = []

        if len(user_past_items) > 0:
            item_idx_current = item_encoder.transform([item_id])[0]
            item_vec_current = item_embeddings[item_idx_current].reshape(1, -1)

            past_indices = item_encoder.transform(user_past_items)
            past_vectors = item_embeddings[past_indices]

            sims = cosine_similarity(item_vec_current, past_vectors)[0]

            top_sim_idx = np.argsort(sims)[::-1][:2]
            similar_items = [user_past_items[j] for j in top_sim_idx]

            similar_explanations = [
                asin_to_title.get(x, x) for x in similar_items
            ]

        explanation = {
            "because_you_liked": similar_explanations,
            "similar_items": [
                asin_to_title.get(x, x)
                for x in get_similar_items(item_id, 2)
            ],
            "predicted_rating": float(predictions[i])
        }

        item_data["explanation"] = explanation

        # ----------- SHAP-style explanation -----------
        u_idx_single = user_id_to_idx[user_id]
        i_idx_single = item_encoder.transform([item_id])[0]

        shap_info = explain_prediction(u_idx_single, i_idx_single)
        item_data["shap_explanation"] = shap_info

        # ----------- Score Breakdown (for controllability UI) -----------
        item_data["score_breakdown"] = {
            "predicted_rating": float(predictions[i]),
            "normalized_rating": float(norm_predictions[i]),
            "diversity": float(get_diversity(item_id)),
            "novelty": float(get_novelty(item_id)),
            "final_score": float(final_scores[i])
        }

        results.append(item_data)

    return {
        "recommendations": results
    }
# def get_top_k_recommendations(user_id, k=10, alpha=0.05, return_scores=True):
#     if user_id not in user_id_to_idx:
#         return []

#     past_items = train_user_items.get(user_id, set())
#     unseen_items = list(all_unique_items - past_items)

#     if len(unseen_items) == 0:
#         return []

#     u_idx = torch.tensor(
#         [user_id_to_idx[user_id]] * len(unseen_items),
#         dtype=torch.long
#     )
#     i_idx = torch.tensor(
#         item_encoder.transform(unseen_items),
#         dtype=torch.long
#     )

#     with torch.no_grad():
#         predictions = model(u_idx, i_idx).numpy()

#     # Top-K selection
#     top_k_idx = np.argpartition(predictions, -k)[-k:]
#     top_k_idx = top_k_idx[np.argsort(predictions[top_k_idx])[::-1]]

#     results = []
#     for i in top_k_idx:
#         item_id = unseen_items[i]

#         item_data = {
#             "item_id": item_id,
#             "name": asin_to_title.get(item_id, "Unknown Product"),
#             "price": asin_to_price.get(item_id, "N/A"),
#             "imageURLHighRes": asin_to_image.get(item_id, "N/A"),
#             "details": asin_to_details.get(item_id, "N/A"),
#             "main_cat": asin_to_main_category.get(item_id, "N/A"),
#             "brand": asin_to_brand.get(item_id, "N/A")
#         }

#         if return_scores:
#             item_data["score"] = float(predictions[i])


#         # Get user's past items
#         user_past_items = list(past_items)

#         similar_explanations = []

#         if len(user_past_items) > 0:
#             item_idx_current = item_encoder.transform([item_id])[0]
#             item_vec_current = item_embeddings[item_idx_current].reshape(1, -1)

#             past_indices = item_encoder.transform(user_past_items)
#             past_vectors = item_embeddings[past_indices]

#             sims = cosine_similarity(item_vec_current, past_vectors)[0]

#             # Get top 2 most similar past items
#             top_sim_idx = np.argsort(sims)[::-1][:2]
#             similar_items = [user_past_items[i] for i in top_sim_idx]

#             similar_explanations = [
#                 asin_to_title.get(x, x) for x in similar_items
#             ]

#         explanation = {
#             "because_you_liked": similar_explanations,
#             "similar_items": [
#                 asin_to_title.get(x, x)
#                 for x in get_similar_items(item_id, 2)
#             ],
#             "predicted_rating": float(predictions[i])
#         }

#         item_data["explanation"] = explanation

#         # SHAP-like breakdown (simple version)
#         u_idx_single = user_id_to_idx[user_id]
#         i_idx_single = item_encoder.transform([item_id])[0]

#         shap_info = explain_prediction(u_idx_single, i_idx_single)
#         item_data["shap_explanation"] = shap_info

#         results.append(item_data)

#     return {
#         "recommendations": results,
#         "model_performance": {
#             "precision": float(np.mean(precisions)),
#             "recall": float(np.mean(recalls)),
#             "f_measure": float(np.mean(f_measures)),
#             "ndcg": float(np.mean(ndcgs))
#         }
#     }

# =========================================================
# ✅ SHAP Style Explanation (FOR API / REUSE)
# We approximate SHAP explanations by decomposing the matrix factorization prediction 
# into additive components (bias terms and latent interaction), which provides local interpretability 
# similar to feature attribution methods.
# =========================================================

def explain_prediction(user_idx, item_idx):
    with torch.no_grad():
        u_vec = model.user_emb.weight[user_idx]
        i_vec = model.item_emb.weight[item_idx]

        user_bias = model.user_bias.weight[user_idx].item()
        item_bias = model.item_bias.weight[item_idx].item()

        interaction = torch.dot(u_vec, i_vec).item()

        pred = global_mean + user_bias + item_bias + interaction

    return {
        "prediction": pred,
        "components": {
            "global_mean": global_mean,
            "user_bias": user_bias,
            "item_bias": item_bias,
            "interaction": interaction
        }
    }

# =========================================================
# Item similarity function
# =========================================================

def get_similar_items(item_id, top_n=3):
    if item_id not in item_encoder.classes_:
        return []

    item_idx = item_encoder.transform([item_id])[0]
    item_vec = item_embeddings[item_idx].reshape(1, -1)

    sims = cosine_similarity(item_vec, item_embeddings)[0]

    top_idx = np.argsort(sims)[::-1][1:top_n+1]
    similar_items = item_encoder.inverse_transform(top_idx)

    return similar_items


# =========================================================
# ✅ EVALUATION
# =========================================================
print("2. Evaluating Top-10 Recommendations...")

precisions, recalls, f_measures, ndcgs = [], [], [], []
num_users_evaluated = 0

with torch.no_grad():
    for user_id, true_test_items in test_user_items.items():

        if len(true_test_items) == 0:
            continue

        top_10_items = [x["item_id"] for x in get_top_k_recommendations(user_id, k=10)["recommendations"]]

        if len(top_10_items) == 0:
            continue

        num_users_evaluated += 1

        hits = len(set(top_10_items) & true_test_items)

        precision = hits / 10.0
        recall = hits / len(true_test_items) if len(true_test_items) > 0 else 0

        f_measure = 0.0
        if (precision + recall) > 0:
            f_measure = 2 * (precision * recall) / (precision + recall)

        # NDCG
        dcg = 0.0
        for rank, item in enumerate(top_10_items):
            if item in true_test_items:
                dcg += 1.0 / math.log2(rank + 2)

        idcg = sum([1.0 / math.log2(i + 2) for i in range(min(10, len(true_test_items)))])
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f_measures.append(f_measure)
        ndcgs.append(ndcg)

# =========================================================
# RESULTS
# =========================================================
print("\n--- Top-10 Recommendation Performance ---")
print(f"Users evaluated: {num_users_evaluated}")
print(f"Precision@10: {np.mean(precisions):.4f}")
print(f"Recall@10:    {np.mean(recalls):.4f}")
print(f"F-measure:    {np.mean(f_measures):.4f}")
print(f"NDCG@10:      {np.mean(ndcgs):.4f}")
print("-----------------------------------------")

def evaluate_model():
    return {
        "users_evaluated": num_users_evaluated,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f_measure": float(np.mean(f_measures)),
        "ndcg": float(np.mean(ndcgs))
    }