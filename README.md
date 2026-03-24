# Project 20 - MLOps Pipeline

End-to-end MLOps pipeline with experiment tracking, model versioning, drift detection, and automated retraining.

## Features

- **Experiment Tracking** — MLflow logs every training run (accuracy, F1, params)
- **Model Versioning** — MLflow Model Registry with version history
- **Drift Detection** — Monitors prediction confidence to detect data drift
- **Auto Retraining** — APScheduler triggers retraining when drift is detected
- **Live Inference** — Test the current model version in real time

## Architecture

```
frontend:3000 → backend:8000 → mlops-service:8001
                                    ↓
                               MLflow (file:./mlruns)
```

## Quick Start

```bash
# mlops-service
cd mlops-service && python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8001 --reload

# backend
cd backend && python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000 --reload

# frontend
cd frontend && npm install && npm start
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/train` | Train a new model run |
| POST | `/api/v1/predict` | Predict sentiment |
| GET | `/api/v1/experiments` | List MLflow runs |
| GET | `/api/v1/versions` | List model versions |
| GET | `/api/v1/metrics` | System metrics |
| GET | `/api/v1/drift` | Drift detection status |

## Tech Stack

- **MLOps**: MLflow 2.10, APScheduler
- **NLP**: scikit-learn (TF-IDF + LogisticRegression)
- **Backend**: FastAPI + httpx
- **Frontend**: React 18 + MUI + Recharts
