# Azure Deployment Guide — Project 20 MLOps Pipeline

---

## Azure Services for MLOps Pipeline

### 1. Ready-to-Use MLOps (No Custom Pipeline Needed)

| Service                              | What it does                                                                 | When to use                                        |
|--------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------|
| **Azure Machine Learning**           | End-to-end MLOps — experiment tracking, model registry, drift detection, pipelines | Replace your MLflow + APScheduler pipeline |
| **Azure ML Data Drift Monitor**      | Automated drift detection on deployed endpoints — replace your manual drift logic | Replace your confidence-based drift detection |
| **Azure ML Pipelines**               | Orchestrate train → evaluate → register → deploy as a DAG                   | Replace your APScheduler auto-retraining           |

> **Azure Machine Learning** is the direct replacement for your MLflow + APScheduler pipeline. Azure ML Experiments replaces MLflow tracking, Model Registry replaces MLflow registry, and Data Drift Monitor replaces your drift detection.

### 2. Host Your Own Pipeline (Keep Current Stack)

| Service                        | What it does                                                        | When to use                                           |
|--------------------------------|---------------------------------------------------------------------|-------------------------------------------------------|
| **Azure Container Apps**       | Run mlops-service + backend containers                              | Best match for your current microservice architecture |
| **Azure Container Registry**   | Store your Docker images                                            | Used with Container Apps or AKS                       |

### 3. Supporting Services

| Service                       | Purpose                                                                  |
|-------------------------------|--------------------------------------------------------------------------|
| **Azure Blob Storage**        | Store training datasets, model artifacts, and MLflow run data            |
| **Azure Key Vault**           | Store API keys and connection strings instead of .env files              |
| **Azure Monitor + App Insights** | Track model accuracy, drift scores, retraining triggers              |

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Azure Static Web Apps — React MLOps Dashboard              │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────────────┐
│  Azure Container Apps — Backend (FastAPI :8000)             │
└──────────────────────┬──────────────────────────────────────┘
                       │ Internal
        ┌──────────────┴──────────────┐
        │ Option A                    │ Option B
        ▼                             ▼
┌───────────────────┐    ┌────────────────────────────────────┐
│ Container Apps    │    │ Azure Machine Learning             │
│ MLOps Service     │    │ Experiments + Pipelines            │
│ :8001             │    │ + Data Drift Monitor + Registry    │
│ MLflow+APScheduler│    │ Full managed MLOps                 │
└───────────────────┘    └────────────────────────────────────┘
```

---

## Prerequisites

```bash
az login
az group create --name rg-mlops-pipeline --location uksouth
az extension add --name containerapp --upgrade
az extension add --name ml --upgrade
```

---

## Step 1 — Create Container Registry and Push Images

```bash
az acr create --resource-group rg-mlops-pipeline --name mlopsacr --sku Basic --admin-enabled true
az acr login --name mlopsacr
ACR=mlopsacr.azurecr.io
docker build -f docker/Dockerfile.mlops-service -t $ACR/mlops-service:latest ./mlops-service
docker push $ACR/mlops-service:latest
docker build -f docker/Dockerfile.backend -t $ACR/backend:latest ./backend
docker push $ACR/backend:latest
```

---

## Step 2 — Create Azure ML Workspace

```bash
az ml workspace create \
  --name mlops-workspace \
  --resource-group rg-mlops-pipeline \
  --location uksouth
```

---

## Step 3 — Deploy Container Apps

```bash
az containerapp env create --name mlops-env --resource-group rg-mlops-pipeline --location uksouth

az containerapp create \
  --name mlops-service --resource-group rg-mlops-pipeline \
  --environment mlops-env --image $ACR/mlops-service:latest \
  --registry-server $ACR --target-port 8001 --ingress internal \
  --min-replicas 1 --max-replicas 3 --cpu 1 --memory 2.0Gi

az containerapp create \
  --name backend --resource-group rg-mlops-pipeline \
  --environment mlops-env --image $ACR/backend:latest \
  --registry-server $ACR --target-port 8000 --ingress external \
  --min-replicas 1 --max-replicas 5 --cpu 0.5 --memory 1.0Gi \
  --env-vars MLOPS_SERVICE_URL=http://mlops-service:8001
```

---

## Option B — Use Azure Machine Learning

```python
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name="rg-mlops-pipeline",
    workspace_name="mlops-workspace"
)

def train_and_register(data_asset_name: str, experiment_name: str) -> str:
    job = command(
        code="./mlops-service",
        command="python train.py",
        environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
        compute="cpu-cluster",
        experiment_name=experiment_name,
        inputs={"training_data": ml_client.data.get(data_asset_name, label="latest")}
    )
    returned_job = ml_client.jobs.create_or_update(job)
    ml_client.jobs.stream(returned_job.name)
    # Register model
    model = Model(path=f"azureml://jobs/{returned_job.name}/outputs/artifacts/paths/model/", name="sentiment-model")
    ml_client.models.create_or_update(model)
    return returned_job.name
```

---

## Estimated Monthly Cost

| Service                  | Tier      | Est. Cost         |
|--------------------------|-----------|-------------------|
| Container Apps (backend) | 0.5 vCPU  | ~$10–15/month     |
| Container Apps (mlops)   | 1 vCPU    | ~$15–20/month     |
| Container Registry       | Basic     | ~$5/month         |
| Static Web Apps          | Free      | $0                |
| Azure ML Workspace       | Basic     | ~$0 (pay per compute) |
| Azure ML Compute (cpu)   | Per hour  | ~$0.096/hour      |
| **Total (Option A)**     |           | **~$30–40/month** |
| **Total (Option B)**     |           | **~$15–20/month + training cost** |

For exact estimates → https://calculator.azure.com

---

## Teardown

```bash
az group delete --name rg-mlops-pipeline --yes --no-wait
```
