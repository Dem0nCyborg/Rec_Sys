import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

print("1. Loading and Encoding Data...")
train_df = pd.read_csv('data/train_data.csv')
test_df = pd.read_csv('data/test_data.csv')

# Neural networks need numbers (0, 1, 2...), not string IDs ("A3CIUO..."). 
# We use LabelEncoder to map every unique User and Item to an integer index.
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

# Fit the encoders on all data so it knows about every user and item
all_users = pd.concat([train_df['reviewerID'], test_df['reviewerID']])
all_items = pd.concat([train_df['asin'], test_df['asin']])
user_encoder.fit(all_users)
item_encoder.fit(all_items)

train_df['user_idx'] = user_encoder.transform(train_df['reviewerID'])
train_df['item_idx'] = item_encoder.transform(train_df['asin'])
test_df['user_idx'] = user_encoder.transform(test_df['reviewerID'])
test_df['item_idx'] = item_encoder.transform(test_df['asin'])

num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
global_mean = train_df['overall'].mean()

# 2. Prepare PyTorch Datasets
class AmazonDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['overall'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

train_loader = DataLoader(AmazonDataset(train_df), batch_size=64, shuffle=True)
test_loader = DataLoader(AmazonDataset(test_df), batch_size=64, shuffle=False)

# 3. Define the Matrix Factorization Model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()
        # These are the "hidden features" the model will learn
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        
        # Biases help the model learn if a user is just generally grumpy or an item is universally loved
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Initialize weights with small random values
        self.user_emb.weight.data.uniform_(-0.05, 0.05)
        self.item_emb.weight.data.uniform_(-0.05, 0.05)
        self.user_bias.weight.data.zero_()
        self.item_bias.weight.data.zero_()

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        
        # The prediction formula: Dot Product + User Bias + Item Bias + Global Mean
        dot_product = (u * i).sum(dim=1)
        prediction = dot_product + self.user_bias(user).squeeze() + self.item_bias(item).squeeze() + global_mean
        return prediction

print("2. Initializing Model and Optimizer...")
model = MatrixFactorization(num_users, num_items, emb_dim=32)
criterion = nn.MSELoss() # Mean Squared Error calculates how far off our stars are
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5) # weight_decay prevents overfitting

print("3. Training the PyTorch Model...")
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for users, items, ratings in train_loader:
        optimizer.zero_grad()
        predictions = model(users, items)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

print("\n4. Evaluating on the Hidden Test Set...")
model.eval()
test_predictions = []
actual_ratings = []

with torch.no_grad():
    for users, items, ratings in test_loader:
        preds = model(users, items)
        test_predictions.extend(preds.tolist())
        actual_ratings.extend(ratings.tolist())

# Bound predictions between 1 and 5 stars (a model might accidentally predict 5.2 stars)
test_predictions = np.clip(test_predictions, 1, 5)

mae = mean_absolute_error(actual_ratings, test_predictions)
rmse = np.sqrt(mean_squared_error(actual_ratings, test_predictions))

print(f"PyTorch MAE:  {mae:.4f}")
print(f"PyTorch RMSE: {rmse:.4f}")

# Save the trained weights so we can use them in the React website later!
torch.save(model.state_dict(), 'data/mf_model_weights.pth')
print("Model weights saved to data/mf_model_weights.pth")