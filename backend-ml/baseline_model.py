import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Loading split datasets...")
train_df = pd.read_csv('data/train_data.csv')
test_df = pd.read_csv('data/test_data.csv')

# 1. Calculate the global average rating from the training data
global_mean = train_df['overall'].mean()
print(f"Global Average Rating in Training Data: {global_mean:.2f} stars")

# 2. The "Dumb" Prediction: Guess the global average for every item in the test set
predictions = [global_mean] * len(test_df)
actual_ratings = test_df['overall']

# 3. Calculate the required Evaluation Metrics
mae = mean_absolute_error(actual_ratings, predictions)
rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))

print("\n--- Baseline Model Evaluation ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print("---------------------------------")
print("Target: Our upcoming PyTorch model must score LOWER than these numbers to be successful.")