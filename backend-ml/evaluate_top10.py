import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import math

print("1. Loading Data and Model...")
train_df = pd.read_csv('data/train_data.csv')
test_df = pd.read_csv('data/test_data.csv')

# Re-create encoders to match the training script exactly
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
all_users = pd.concat([train_df['reviewerID'], test_df['reviewerID']])
all_items = pd.concat([train_df['asin'], test_df['asin']])
user_encoder.fit(all_users)
item_encoder.fit(all_items)

num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
global_mean = train_df['overall'].mean()

# Re-define the architecture so PyTorch knows how to load the weights
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

# Load the trained "brain" you just generated
model = MatrixFactorization(num_users, num_items, emb_dim=32)
model.load_state_dict(torch.load('data/mf_model_weights_BEST_0.0939.pth'))
model.eval() # Set to evaluation mode

print("2. Generating Top-10 Recommendations for each user...")
# Group data for fast lookup
train_user_items = train_df.groupby('reviewerID')['asin'].apply(set).to_dict()
test_user_items = test_df.groupby('reviewerID')['asin'].apply(set).to_dict()
all_unique_items = set(all_items)

precisions, recalls, f_measures, ndcgs = [], [], [], []

# Evaluate the Top 10 for every user who is in the test set
with torch.no_grad():
    for user_id, true_test_items in test_user_items.items():
        # 1. Find items the user has NEVER purchased in the training set
        past_items = train_user_items.get(user_id, set())
        unseen_items = list(all_unique_items - past_items)
        
        # 2. Encode for the model
        u_idx = torch.tensor([user_encoder.transform([user_id])[0]] * len(unseen_items), dtype=torch.long)
        i_idx = torch.tensor(item_encoder.transform(unseen_items), dtype=torch.long)
        
        # 3. Predict all unseen items at once
        predictions = model(u_idx, i_idx).numpy()
        
        # 4. Rank and get the Top 10 Remf_model_weights.pthcommended Items
        top_10_indices = np.argsort(predictions)[::-1][:10]
        top_10_items = [unseen_items[idx] for idx in top_10_indices]
        
        # 5. Calculate Metrics
        hits = len(set(top_10_items) & true_test_items)
        
        precision = hits / 10.0
        recall = hits / len(true_test_items)
        
        f_measure = 0.0
        if (precision + recall) > 0:
            f_measure = 2 * (precision * recall) / (precision + recall)
            
        # Calculate NDCG
        dcg = 0.0
        for i, item in enumerate(top_10_items):
            if item in true_test_items:
                dcg += 1.0 / math.log2(i + 2) # i+2 because index is 0-based and formula needs log2(rank+1)
                
        idcg = sum([1.0 / math.log2(i + 2) for i in range(min(10, len(true_test_items)))])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f_measures.append(f_measure)
        ndcgs.append(ndcg)

print("\n--- Top-10 Recommendation Performance ---")
print(f"Precision@10: {np.mean(precisions):.4f}")
print(f"Recall@10:    {np.mean(recalls):.4f}")
print(f"F-measure:    {np.mean(f_measures):.4f}")
print(f"NDCG@10:      {np.mean(ndcgs):.4f}")
print("-----------------------------------------")