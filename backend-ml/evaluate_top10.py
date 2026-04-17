import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import math
import json
import gzip


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
# =========================================================
def get_top_k_recommendations(user_id, k=10, return_scores=True):
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

    # Top-K selection
    top_k_idx = np.argpartition(predictions, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(predictions[top_k_idx])[::-1]]

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

        if return_scores:
            item_data["score"] = float(predictions[i])

        results.append(item_data)

    return {
        "recommendations": results,
        "model_performance": {
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f_measure": float(np.mean(f_measures)),
            "ndcg": float(np.mean(ndcgs))
        }
    }

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