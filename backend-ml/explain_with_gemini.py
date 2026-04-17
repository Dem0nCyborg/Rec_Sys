import pandas as pd
import os
from google import genai

# 1. Setup Gemini
client = genai.Client()

# 2. Load Data
print("Loading datasets...")
df = pd.read_json('data/All_Beauty_5.json.gz', lines=True, compression='gzip')
# Assuming evaluate_top10.py saved a CSV called 'top10_recommendations.csv'
# If you don't have the CSV yet, we'll pick one user and their top 10 from the MF model
recs_df = pd.read_csv('data/test_data.csv') # Just to get a valid user for the demo

def get_personalized_reason(user_id, item_id):
    # (Same data loading as before)
    user_past = df[df['reviewerID'] == user_id].sort_values('overall', ascending=False).head(3)
    user_context = " | ".join(user_past['reviewText'].astype(str))

    item_info = df[df['asin'] == item_id].head(5)
    item_context = " | ".join(item_info['reviewText'].astype(str))

    prompt = f"""
    User Preference Context: "{user_context}"
    Recommended Product Reviews: "{item_context}"
    
    Using the Gemma 3 reasoning style, explain in ONE conversational sentence 
    why this specific product matches the user's past behavior.
    """
    
    # CHANGE THIS LINE TO USE GEMMA 3 27B
    response = client.models.generate_content(
        model="gemma-3-27b-it", 
        contents=prompt
    )
    return response.text.strip()

# --- GENERATE FOR TOP 10 ---
target_user = "A281N877SNCA7H" # Example User ID from Beauty
# For this demo, we'll pick 5 random items to represent the 'Top Recommendations'
top_items = df['asin'].unique()[:5] 

print(f"\n--- PERSONALIZED RECOMMENDATIONS FOR {target_user} ---")
for i, item_id in enumerate(top_items, 1):
    reason = get_personalized_reason(target_user, item_id)
    print(f"{i}. Product ID: {item_id}")
    print(f"   Why: {reason}\n")