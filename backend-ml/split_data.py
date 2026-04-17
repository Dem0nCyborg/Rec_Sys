import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading dataset for splitting...")
file_path = "data/All_Beauty_5.json.gz" 
df = pd.read_json(file_path, lines=True, compression='gzip')
#df = df.groupby('reviewerID').filter(lambda x: len(x) >= 5)

# Keep only the essential columns to save memory
df = df[['reviewerID', 'asin', 'overall', 'reviewText']]

print("Splitting data 80/20 for each user...")
train_list = []
test_list = []

# Group by each unique user
for user_id, group in df.groupby('reviewerID'):
    # Since it's a 5-core dataset, every user has at least 5 reviews.
    # We take 20% for testing and 80% for training for EACH user.
    # random_state=42 ensures we get the exact same split if we run this again
    train, test = train_test_split(group, test_size=0.2, random_state=42)
    
    train_list.append(train)
    test_list.append(test)

# Combine all the individual splits back into two master dataframes
train_df = pd.concat(train_list)
test_df = pd.concat(test_list)

print(f"Training set size: {len(train_df)} reviews (80%)")
print(f"Testing set size: {len(test_df)} reviews (20%)")

# Save the results to CSV files so our model can use them later
print("Saving to CSV...")
train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)

print("Done! Data is ready for machine learning.")