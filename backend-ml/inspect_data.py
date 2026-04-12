import pandas as pd

print("Loading Amazon dataset...")



file_path = "data/All_Beauty_5.json.gz" 

# Reading the compressed JSON Lines file into a DataFrame
df = pd.read_json(file_path, lines=True, compression='gzip')

print(f"Success! Loaded {len(df)} reviews.\n")
print("Here are the first 5 rows of the core columns:")

# Only isolating the columns we need for the Recommender System
print(df[['reviewerID', 'asin', 'overall', 'reviewText']].head())