"""
main.py  —  FastAPI backend with sentiment-aware recommendations
"""

import re
import os
import json

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from dotenv import load_dotenv
from google import genai

from evaluate_top10 import evaluate_model, get_top_k_recommendations

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Gemini client ──────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Loading dataset for backend...")
df = pd.read_json('data/All_Beauty_5.json.gz', lines=True, compression='gzip')

# ── Evaluation (runs once at startup) ─────────────────────────────────────────
model_metrics = evaluate_model()


# ── Helpers ────────────────────────────────────────────────────────────────────
def safe_parse_json(text: str):
    try:
        cleaned = re.sub(r"```json|```", "", text).strip()
        match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(0))
    except Exception as e:
        print("JSON parse error:", e)
        return None


def generate_explanation(user_context: str, item_name: str, item_brand: str = "",
                         sentiment_label: str = ""):
    sentiment_hint = (
        f"\nNote: Customer reviews for this product are {sentiment_label.lower()}."
        if sentiment_label else ""
    )
    prompt = f"""You are a recommendation explanation generator.

User history:
{user_context}

Item:
Name: {item_name}
Brand: {item_brand}{sentiment_hint}

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
            contents=prompt,
        )
        parsed = safe_parse_json(response.text)
        if parsed:
            return parsed
    except Exception:
        pass

    return {
        "headline": "Recommended for you",
        "why_recommended": ["Matches your preferences"],
        "score_explanation": "This item aligns with your behavior.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/recommend/{user_id}")
async def get_recommendations(
    user_id: str,
    w1:    float = Query(1.0, description="Weight for NCF accuracy"),
    w2:    float = Query(0.0, description="Weight for diversity"),
    w3:    float = Query(0.0, description="Weight for novelty"),
    alpha: float = Query(0.0, description="Weight for sentiment (0=off, 0.3=recommended)"),
):
    """
    Return Top-10 recommendations for a user.

    Query parameters
    ----------------
    w1    : float — NCF accuracy weight   (default 1.0)
    w2    : float — diversity weight      (default 0.0)
    w3    : float — novelty weight        (default 0.0)
    alpha : float — sentiment re-ranking  (default 0.0)

    Example:
        GET /recommend/A1XYZ?w1=0.7&alpha=0.3
    """
    result          = get_top_k_recommendations(user_id, k=10, w1=w1, w2=w2, w3=w3, alpha=alpha)
    recommendations = result.get("recommendations", [])

    return {
        "user_id":          user_id,
        "alpha_used":       alpha,
        "recommendations":  recommendations,
        "model_performance": model_metrics,
    }


@app.get("/metrics")
async def get_metrics():
    """
    Return pre-computed evaluation metrics:
      - baseline (alpha=0, pure NCF)
      - best alpha found during sweep
      - full alpha sweep table
    """
    return model_metrics


@app.get("/sentiment/{asin}")
async def get_product_sentiment(asin: str):
    """Return raw and normalised VADER sentiment for a single product."""
    from evaluate_top10 import _raw_sentiment_scores, sentiment_scores_norm, _sentiment_label
    raw  = _raw_sentiment_scores.get(asin)
    norm = sentiment_scores_norm.get(asin)
    if raw is None:
        return {"asin": asin, "error": "Product not found in sentiment index"}
    return {
        "asin":            asin,
        "sentiment_score": round(raw, 4),
        "sentiment_norm":  round(norm, 4),
        "label":           _sentiment_label(raw),
    }
