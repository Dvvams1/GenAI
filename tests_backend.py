import pytest
from fastapi.testclient import TestClient
from backend import app

client = TestClient(app)

def test_analyze_review():
    response = client.post("/api/analyze", json={"review": "The food was great but the service was slow."})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert data["sentiment"] in ["positive", "negative", "neutral"]
    assert "emoji" in data
    assert "score" in data
    assert "aspects" in data
    assert "food" in data["aspects"]
    assert "hygiene" in data["aspects"]
    assert "service" in data["aspects"]
