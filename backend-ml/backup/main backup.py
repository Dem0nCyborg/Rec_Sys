import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from evaluate_top10 import evaluate_model, get_top_k_recommendations
import os
from fastapi import Query
from google import genai
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow the React frontend to talk to this Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize Gemma & Load Data
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
print("Loading dataset for backend...")
df = pd.read_json('data/All_Beauty_5.json.gz', lines=True, compression='gzip')

def simplify_shap(shap):
    c = shap.get("components", {})

    factors = []

    if c.get("global_mean", 0) > 4.5:
        factors.append("High quality product category")

    if c.get("user_bias", 0) > 0:
        factors.append("Strong match with your past behavior")

    if c.get("item_bias", 0) > 0:
        factors.append("Item is stronger than average alternatives")

    if c.get("interaction", 0) > 0.1:
        factors.append("Very strong user-item compatibility")

    return factors

def generate_explanation(user_context: str, item_name: str, item_brand: str = ""):
    prompt = f"""You are a recommendation explanation generator.

    User history:
    {user_context}

    Item:
    Name: {item_name}
    Brand: {item_brand}

    Return ONLY valid JSON:

    {{
    "headline": "string",
    "why_recommended": ["string", "string", "string"],
    "score_explanation": "string"
    }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        print(response)
        print("LLM response:", response.text)  # Debugging line to see raw LLM output

        parsed = safe_parse_json(response.text)

        if parsed:
            return parsed

        return {
            "headline": "Recommended for you",
            "why_recommended": ["Matches your preferences"],
            "score_explanation": "This item aligns with your behavior."
        }

    except Exception:
        return {
            "headline": "Recommended for you",
            "why_recommended": ["Matches your preferences"],
            "score_explanation": "This item aligns with your behavior."
        }
    

def safe_parse_json(text: str):
    try:
        # Remove markdown code blocks
        cleaned = re.sub(r"```json|```", "", text).strip()

        # Extract only the first JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None

        json_str = match.group(0)

        return json.loads(json_str)

    except Exception as e:
        print("JSON parse error:", e)
        return None
    

model_metrics = evaluate_model()


@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: str,
    w1: float = Query(1.0, description="Weight for accuracy"),
    w2: float = Query(0.0, description="Weight for diversity"),
    w3: float = Query(0.0, description="Weight for novelty")
):
    # 2. Mocking the Top 3 items from your PyTorch model
    result = get_top_k_recommendations(user_id, 10, w1=w1, w2=w2, w3=w3)
    recommendations = result.get("recommendations", []) # getting top-10 recommendations for the user
    model_performance = model_metrics if model_metrics else {} # getting model performance metrics (optional)


    # 3. Get User Context
    # user_past = df[df['reviewerID'] == user_id].sort_values('overall', ascending=False).head(3)
    # user_context = " | ".join(user_past['reviewText'].astype(str))

    # # generate Gemini explanations for each recommended item using SHAP and LLM 
    # enriched = []

    # for rec in recommendations:

    #     try:
    #         item_row = df[df['asin'] == rec["item_id"]].head(1)

    #         item_name = rec.get("name", "")
    #         item_brand = item_row["brand"].values[0] if "brand" in item_row else ""


    #         # LLM explanation
    #         explanation = generate_explanation(
    #             user_context,
    #             item_name,
    #             item_brand
    #         )

    #         enriched.append({
    #             "item_id": rec["item_id"],
    #             "name": item_name,
    #             "score": rec.get("score"),
    #             "shap_factors": shap_factors,
    #             "explanation": explanation
    #         })

    #     except Exception:
    #         enriched.append({
    #             "item_id": rec.get("item_id"),
    #             "name": rec.get("name"),
    #             "score": rec.get("score"),
    #             "shap_factors": [],
    #             "explanation": {
    #                 "headline": "Recommended for you",
    #                 "why_recommended": ["Matches your preferences"],
    #                 "score_explanation": "This item fits your profile."
    #             }
    #         })

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "model_performance": model_performance
        # "enriched": enriched
    }