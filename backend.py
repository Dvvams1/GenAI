from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Optional
import os
from pathlib import Path

# Sentiment analysis dependencies
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from textblob import TextBlob

app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory=str(Path(__file__).parent))

# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

class ReviewRequest(BaseModel):
    review: str

class ReviewResponse(BaseModel):
    sentiment: str
    emoji: str
    score: float
    aspects: Dict[str, str]

# Basic rule-based aspect extractor
ASPECT_KEYWORDS = {
    "food": ["food", "taste", "flavor", "dish", "meal", "delicious", "eat"],
    "hygiene": ["clean", "hygiene", "dirty", "sanitary", "smell", "wash", "bathroom"],
    "service": ["service", "waiter", "staff", "friendly", "rude", "slow", "quick"]
}

# Sentiment label and emoji mapping
SENTIMENT_MAP = {
    "positive": "❤️",
    "negative": "😡",
    "neutral": "😐"
}

# Initialize sentiment pipeline if HF is available
if HF_AVAILABLE:
    sentiment_analyzer = pipeline("sentiment-analysis")


def analyze_sentiment_hf(text: str):
    # Use HF pipeline
    result = sentiment_analyzer(text)[0]
    label = result['label'].lower()
    score = float(result['score'])
    if label not in SENTIMENT_MAP:
        label = "neutral"
    return label, score


def analyze_sentiment_textblob(text: str):
    # Use TextBlob polarity
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive", polarity
    elif polarity < -0.1:
        return "negative", -polarity
    else:
        return "neutral", abs(polarity)


def extract_aspects(text: str) -> Dict[str, str]:
    text_lower = text.lower()
    aspects = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        # Basic counting of positive and negative words
        pos_count = 0
        neg_count = 0
        for kw in keywords:
            if kw in text_lower:
                # Simple heuristic: if keyword present, count sentiment of sentence containing it
                # For simplicity, use TextBlob polarity of whole text
                blob = TextBlob(text_lower)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    pos_count += 1
                elif polarity < -0.1:
                    neg_count += 1
        if pos_count > neg_count:
            aspects[aspect] = "positive"
        elif neg_count > pos_count:
            aspects[aspect] = "negative"
        else:
            aspects[aspect] = "neutral"
    return aspects


@app.post("/api/analyze", response_model=ReviewResponse)
async def analyze_review(request: ReviewRequest):
    review = request.review.strip()
    if not review:
        raise HTTPException(status_code=400, detail="Review text is empty")

    if HF_AVAILABLE:
        sentiment, score = analyze_sentiment_hf(review)
    else:
        sentiment, score = analyze_sentiment_textblob(review)

    emoji = SENTIMENT_MAP.get(sentiment, "😐")
    aspects = extract_aspects(review)

    return ReviewResponse(sentiment=sentiment, emoji=emoji, score=round(score, 2), aspects=aspects)

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return FileResponse('frontend.html')

# Handle all other routes to serve the frontend
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    if os.path.exists(full_path):
        return FileResponse(full_path)
    return FileResponse('frontend.html')

# Add CORS middleware to allow frontend access
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
