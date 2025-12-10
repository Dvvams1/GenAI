# Review Sentiment Analyzer Web App

## Overview
This is a simple web application to analyze the sentiment of a single text review instantly. It uses a FastAPI backend with a lightweight sentiment analysis pipeline and a vanilla JavaScript frontend.

---

## Features
- Textarea input for a review
- "Analyze Review" button
- Live result card showing sentiment text (Positive / Negative / Neutral), emoji output (❤️ 😡 😐), and confidence score
- Basic rule-based aspect extraction for food, hygiene, and service
- Small animated transition when the result card appears

---

## Tech Stack
- Frontend: HTML, CSS, vanilla JavaScript
- Backend: FastAPI (Python)
- Sentiment Analysis: HuggingFace transformers (if available), fallback to TextBlob

---

## Setup & Run Locally

### Prerequisites
- Python 3.8+
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the backend server
```bash
uvicorn backend:app --reload
```

### Open the frontend
Open `frontend.html` in your browser (e.g., drag and drop or use a local HTTP server).

---

## API

### POST /api/analyze
Request JSON:
```json
{
  "review": "Your review text here"
}
```

Response JSON:
```json
{
  "sentiment": "positive" | "negative" | "neutral",
  "emoji": "❤️" | "😡" | "😐",
  "score": 0.92,
  "aspects": {
    "food": "positive",
    "hygiene": "neutral",
    "service": "negative"
  }
}
```

---

## Configuration

### HuggingFace API Key (optional)
If you want to use HuggingFace transformers and your model requires authentication, set the environment variable:
```bash
export HF_API_KEY=your_api_key_here
```

---

## Deployment

### Vercel
- Vercel primarily supports frontend frameworks. For backend, consider using Serverless Functions or deploy backend separately.

### Render
- Create a new Web Service on Render.
- Connect your repo.
- Set the start command:
  ```
  uvicorn backend:app --host 0.0.0.0 --port 10000
  ```
- Set environment variables if needed.

### Heroku
- Create a new app.
- Push your code.
- Heroku will use the `Procfile` to start the app.
- Set environment variables via Heroku dashboard or CLI.

---

## Testing
Run the backend test with:
```bash
pytest tests_backend.py
```

---

## Notes
- Frontend and backend are separate; frontend is static and can be served by any HTTP server.
- Backend requires Python environment.
