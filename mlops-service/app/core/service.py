import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import os
from datetime import datetime
import numpy as np
from .config import settings
from .dataset import prepare_data

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(settings.EXPERIMENT_NAME)

_current_model = None
_current_version = None
_training_history = []
_prediction_log = []

def train_model(params=None):
    global _current_model, _current_version
    
    if params is None:
        params = {"C": 1.0, "max_iter": 100}
    
    X_train, X_test, y_train, y_test = prepare_data()
    
    with mlflow.start_run() as run:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=1000)),
            ("clf", LogisticRegression(**params))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
        
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(pipeline, "model", registered_model_name=settings.MODEL_REGISTRY_NAME)
        
        _current_model = pipeline
        _current_version = run.info.run_id
        
        _training_history.append({
            "run_id": run.info.run_id,
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "params": params
        })
        
        return {
            "run_id": run.info.run_id,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

def get_current_model():
    global _current_model
    if _current_model is None:
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{settings.MODEL_REGISTRY_NAME}'")
            if versions:
                latest = max(versions, key=lambda v: v.creation_timestamp)
                _current_model = mlflow.sklearn.load_model(f"models:/{settings.MODEL_REGISTRY_NAME}/{latest.version}")
        except:
            pass
    return _current_model

def predict(text):
    model = get_current_model()
    if model is None:
        raise ValueError("No model available")
    
    prediction = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = float(max(proba))
    
    _prediction_log.append({
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "prediction": prediction,
        "confidence": confidence
    })
    
    return {"prediction": prediction, "confidence": confidence}

def get_experiments():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(settings.EXPERIMENT_NAME)
    if not experiment:
        return []
    
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=10)
    return [{
        "run_id": run.info.run_id,
        "start_time": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
        "metrics": run.data.metrics,
        "params": run.data.params,
        "status": run.info.status
    } for run in runs]

def get_model_versions():
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{settings.MODEL_REGISTRY_NAME}'")
        return [{
            "version": v.version,
            "run_id": v.run_id,
            "creation_timestamp": datetime.fromtimestamp(v.creation_timestamp / 1000).isoformat(),
            "current_stage": v.current_stage
        } for v in versions]
    except:
        return []

def get_metrics():
    recent_predictions = _prediction_log[-100:] if len(_prediction_log) > 0 else []
    avg_confidence = np.mean([p["confidence"] for p in recent_predictions]) if recent_predictions else 0.0
    
    return {
        "total_predictions": len(_prediction_log),
        "total_trainings": len(_training_history),
        "current_version": _current_version,
        "avg_confidence": float(avg_confidence),
        "recent_predictions": len(recent_predictions)
    }

def check_drift():
    if len(_prediction_log) < 10:
        return {"drift_detected": False, "drift_score": 0.0}
    
    recent = _prediction_log[-50:]
    avg_confidence = np.mean([p["confidence"] for p in recent])
    drift_score = 1.0 - avg_confidence
    
    return {
        "drift_detected": drift_score > settings.DRIFT_THRESHOLD,
        "drift_score": float(drift_score),
        "threshold": settings.DRIFT_THRESHOLD
    }
