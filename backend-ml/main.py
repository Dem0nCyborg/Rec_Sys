from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from evaluate_top10 import get_top_k_recommendations
# from google import genai
import os


app = FastAPI()

# Allow the React frontend to talk to this Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize Gemma & Load Data
# client = genai.Client()
print("Loading dataset for backend...")
df = pd.read_json('data/All_Beauty_5.json.gz', lines=True, compression='gzip')

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: str):
    # 2. Mocking the Top 3 items from your PyTorch model
    result = get_top_k_recommendations(user_id, 10)
    recommendations = result.get("recommendations", []) # getting top-10 recommendations for the user
    model_performance = result.get("model_performance", {}) # getting model performance metrics (optional)


    # 3. Get User Context
    user_past = df[df['reviewerID'] == user_id].sort_values('overall', ascending=False).head(3)
    user_context = " | ".join(user_past['reviewText'].astype(str))

    # 4. Ask Gemma 3 for the explanation of the first product
    try:
        item_info = df[df['asin'] == recommendations[0]["id"]].head(3)
        item_context = " | ".join(item_info['reviewText'].astype(str))
        
        prompt = f"User likes: {user_context}. Item reviews: {item_context}. Explain in ONE conversational sentence why this specific product matches the user's past behavior."
        
        # response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
        # ai_reason = response.text.strip()
        ai_reason = ""
    except Exception as e:
        ai_reason = "Based on your preference for high-quality items, this matches your profile." # Fallback if API rate limit hits

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "model_performance": model_performance,
        "ai_reason": ai_reason
    }