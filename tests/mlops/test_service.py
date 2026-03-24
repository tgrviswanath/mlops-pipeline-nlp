import pytest
from unittest.mock import patch, MagicMock
from app.core.service import train_model, predict, check_drift, _prediction_log

def test_train_returns_metrics():
    result = train_model({"C": 1.0, "max_iter": 10})
    assert "accuracy" in result
    assert "run_id" in result
    assert 0.0 <= result["accuracy"] <= 1.0

def test_predict_after_train():
    train_model({"C": 1.0, "max_iter": 10})
    result = predict("This is great!")
    assert result["prediction"] in ["positive", "negative", "neutral"]
    assert 0.0 <= result["confidence"] <= 1.0

def test_predict_logs_prediction():
    train_model({"C": 1.0, "max_iter": 10})
    before = len(_prediction_log)
    predict("Test text")
    assert len(_prediction_log) == before + 1

def test_drift_no_data():
    result = check_drift()
    assert result["drift_detected"] is False
    assert result["drift_score"] == 0.0

def test_train_multiple_runs():
    r1 = train_model({"C": 0.5, "max_iter": 10})
    r2 = train_model({"C": 2.0, "max_iter": 10})
    assert r1["run_id"] != r2["run_id"]
