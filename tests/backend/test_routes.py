import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@patch("app.api.routes.call", new_callable=AsyncMock)
def test_train(mock_call):
    mock_call.return_value = {"run_id": "abc123", "accuracy": 0.95}
    r = client.post("/api/v1/train", json={"params": {"C": 1.0, "max_iter": 100}})
    assert r.status_code == 200
    assert r.json()["accuracy"] == 0.95

@patch("app.api.routes.call", new_callable=AsyncMock)
def test_predict(mock_call):
    mock_call.return_value = {"prediction": "positive", "confidence": 0.92}
    r = client.post("/api/v1/predict", json={"text": "Great product!"})
    assert r.status_code == 200
    assert r.json()["prediction"] == "positive"

@patch("app.api.routes.call", new_callable=AsyncMock)
def test_metrics(mock_call):
    mock_call.return_value = {"total_predictions": 10, "avg_confidence": 0.88}
    r = client.get("/api/v1/metrics")
    assert r.status_code == 200

@patch("app.api.routes.call", new_callable=AsyncMock)
def test_drift(mock_call):
    mock_call.return_value = {"drift_detected": False, "drift_score": 0.05}
    r = client.get("/api/v1/drift")
    assert r.status_code == 200
    assert r.json()["drift_detected"] is False
