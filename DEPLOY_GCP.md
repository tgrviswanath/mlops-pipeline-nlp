# GCP Deployment Guide — Project 20 MLOps Pipeline

---

## GCP Services for MLOps Pipeline

### 1. Ready-to-Use MLOps (No Custom Pipeline Needed)

| Service                              | What it does                                                                 | When to use                                        |
|--------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------|
| **Vertex AI**                        | End-to-end MLOps — experiment tracking, model registry, drift detection, pipelines | Replace your MLflow + APScheduler pipeline |
| **Vertex AI Model Monitoring**       | Automated drift detection on deployed endpoints — replace your manual drift logic | Replace your confidence-based drift detection |
| **Vertex AI Pipelines**              | Orchestrate train → evaluate → register → deploy as a Kubeflow DAG          | Replace your APScheduler auto-retraining           |

> **Vertex AI** is the direct replacement for your MLflow + APScheduler pipeline. Vertex AI Experiments replaces MLflow tracking, Model Registry replaces MLflow registry, and Model Monitoring replaces your drift detection.

### 2. Host Your Own Pipeline (Keep Current Stack)

| Service                    | What it does                                                        | When to use                                           |
|----------------------------|---------------------------------------------------------------------|-------------------------------------------------------|
| **Cloud Run**              | Run mlops-service + backend containers — serverless, scales to zero | Best match for your current microservice architecture |
| **Artifact Registry**      | Store your Docker images                                            | Used with Cloud Run or GKE                            |

### 3. Supporting Services

| Service                        | Purpose                                                                   |
|--------------------------------|---------------------------------------------------------------------------|
| **Cloud Storage**              | Store training datasets, model artifacts, and MLflow run data             |
| **Secret Manager**             | Store API keys and connection strings instead of .env files               |
| **Cloud Monitoring + Logging** | Track model accuracy, drift scores, retraining triggers                   |

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Firebase Hosting — React MLOps Dashboard                   │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────────────┐
│  Cloud Run — Backend (FastAPI :8000)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │ Internal HTTPS
        ┌──────────────┴──────────────┐
        │ Option A                    │ Option B
        ▼                             ▼
┌───────────────────┐    ┌────────────────────────────────────┐
│ Cloud Run         │    │ Vertex AI                          │
│ MLOps Service     │    │ Experiments + Pipelines            │
│ :8001             │    │ + Model Monitoring + Registry      │
│ MLflow+APScheduler│    │ Full managed MLOps                 │
└───────────────────┘    └────────────────────────────────────┘
```

---

## Prerequisites

```bash
gcloud auth login
gcloud projects create mlops-project --name="MLOps Pipeline"
gcloud config set project mlops-project
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com aiplatform.googleapis.com \
  storage.googleapis.com cloudbuild.googleapis.com
```

---

## Step 1 — Create Artifact Registry and Push Images

```bash
GCP_REGION=europe-west2
gcloud artifacts repositories create mlops-repo \
  --repository-format=docker --location=$GCP_REGION
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev
AR=$GCP_REGION-docker.pkg.dev/mlops-project/mlops-repo
docker build -f docker/Dockerfile.mlops-service -t $AR/mlops-service:latest ./mlops-service
docker push $AR/mlops-service:latest
docker build -f docker/Dockerfile.backend -t $AR/backend:latest ./backend
docker push $AR/backend:latest
```

---

## Step 2 — Create Cloud Storage for MLflow Artifacts

```bash
gsutil mb -l $GCP_REGION gs://mlops-artifacts-mlops-project
```

---

## Step 3 — Deploy to Cloud Run

```bash
gcloud run deploy mlops-service \
  --image $AR/mlops-service:latest --region $GCP_REGION \
  --port 8001 --no-allow-unauthenticated \
  --min-instances 1 --max-instances 3 --memory 2Gi --cpu 1

MLOPS_URL=$(gcloud run services describe mlops-service --region $GCP_REGION --format "value(status.url)")

gcloud run deploy backend \
  --image $AR/backend:latest --region $GCP_REGION \
  --port 8000 --allow-unauthenticated \
  --min-instances 1 --max-instances 5 --memory 1Gi --cpu 1 \
  --set-env-vars MLOPS_SERVICE_URL=$MLOPS_URL
```

---

## Option B — Use Vertex AI for MLOps

```python
import vertexai
from vertexai.preview import experiments
from google.cloud import aiplatform

vertexai.init(project="mlops-project", location="europe-west2")

def train_and_register(gcs_data_uri: str, experiment_name: str) -> str:
    aiplatform.init(experiment=experiment_name)
    with aiplatform.start_run(run=f"run-{experiment_name}"):
        # Log params
        aiplatform.log_params({"max_iter": 1000, "model": "LogisticRegression"})
        # Submit training job
        job = aiplatform.CustomTrainingJob(
            display_name=experiment_name,
            script_path="./mlops-service/train.py",
            container_uri="europe-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-0:latest"
        )
        model = job.run(
            dataset=aiplatform.TabularDataset.create(gcs_source=gcs_data_uri),
            model_display_name="sentiment-model",
            machine_type="n1-standard-4"
        )
        aiplatform.log_metrics({"accuracy": 0.85})
    return model.resource_name
```

---

## Estimated Monthly Cost

| Service                    | Tier                  | Est. Cost          |
|----------------------------|-----------------------|--------------------|
| Cloud Run (backend)        | 1 vCPU / 1 GB         | ~$10–15/month      |
| Cloud Run (mlops-service)  | 1 vCPU / 2 GB         | ~$12–18/month      |
| Artifact Registry          | Storage               | ~$1–2/month        |
| Firebase Hosting           | Free tier             | $0                 |
| Cloud Storage              | Standard              | ~$2–5/month        |
| Vertex AI Training         | Per node hour         | ~$0.38/hour        |
| **Total (Option A)**       |                       | **~$25–40/month**  |
| **Total (Option B)**       |                       | **~$13–22/month + training cost** |

For exact estimates → https://cloud.google.com/products/calculator

---

## Teardown

```bash
gcloud run services delete backend --region $GCP_REGION --quiet
gcloud run services delete mlops-service --region $GCP_REGION --quiet
gcloud artifacts repositories delete mlops-repo --location=$GCP_REGION --quiet
gsutil rm -r gs://mlops-artifacts-mlops-project
gcloud projects delete mlops-project
```
