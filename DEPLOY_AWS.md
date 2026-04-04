# AWS Deployment Guide — Project 20 MLOps Pipeline

---

## AWS Services for MLOps Pipeline

### 1. Ready-to-Use MLOps (No Custom Pipeline Needed)

| Service                    | What it does                                                                 | When to use                                        |
|----------------------------|------------------------------------------------------------------------------|----------------------------------------------------|
| **AWS SageMaker**          | End-to-end MLOps — experiment tracking, model registry, drift detection, auto-retraining | Replace your MLflow + APScheduler pipeline |
| **SageMaker Model Monitor**| Automated drift detection on deployed endpoints — replace your manual drift logic | Replace your confidence-based drift detection |
| **SageMaker Pipelines**    | Orchestrate train → evaluate → register → deploy as a DAG                   | Replace your APScheduler auto-retraining           |

> **AWS SageMaker** is the direct replacement for your MLflow + APScheduler pipeline. SageMaker Experiments replaces MLflow tracking, Model Registry replaces MLflow registry, and Model Monitor replaces your drift detection.

### 2. Host Your Own Pipeline (Keep Current Stack)

| Service                    | What it does                                                        | When to use                                           |
|----------------------------|---------------------------------------------------------------------|-------------------------------------------------------|
| **AWS App Runner**         | Run mlops-service + backend containers                              | Quickest path to production                           |
| **Amazon ECS Fargate**     | Run mlops-service + backend containers in a private VPC             | Best match for your current microservice architecture |
| **Amazon ECR**             | Store your Docker images                                            | Used with App Runner, ECS, or EKS                     |

### 3. Supporting Services

| Service                  | Purpose                                                                   |
|--------------------------|---------------------------------------------------------------------------|
| **Amazon S3**            | Store training datasets, model artifacts, and MLflow run data             |
| **Amazon DynamoDB**      | Store experiment metadata and drift detection results                     |
| **AWS Secrets Manager**  | Store API keys and connection strings instead of .env files               |
| **Amazon CloudWatch**    | Track model accuracy, drift scores, retraining triggers                   |

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  S3 + CloudFront — React MLOps Dashboard Frontend           │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────────────┐
│  AWS App Runner — Backend (FastAPI :8000)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │ Internal
        ┌──────────────┴──────────────┐
        │ Option A                    │ Option B
        ▼                             ▼
┌───────────────────┐    ┌────────────────────────────────────┐
│ ECS Fargate       │    │ AWS SageMaker                      │
│ MLOps Service     │    │ Experiments + Pipelines            │
│ :8001             │    │ + Model Monitor + Registry         │
│ MLflow+APScheduler│    │ Full managed MLOps                 │
└───────────────────┘    └────────────────────────────────────┘
```

---

## Prerequisites

```bash
aws configure
AWS_REGION=eu-west-2
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
```

---

## Step 1 — Create ECR and Push Images

```bash
aws ecr create-repository --repository-name mlops/mlops-service --region $AWS_REGION
aws ecr create-repository --repository-name mlops/backend --region $AWS_REGION
ECR=$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR
docker build -f docker/Dockerfile.mlops-service -t $ECR/mlops/mlops-service:latest ./mlops-service
docker push $ECR/mlops/mlops-service:latest
docker build -f docker/Dockerfile.backend -t $ECR/mlops/backend:latest ./backend
docker push $ECR/mlops/backend:latest
```

---

## Step 2 — Create S3 Bucket for MLflow Artifacts

```bash
aws s3 mb s3://mlops-artifacts-$AWS_ACCOUNT --region $AWS_REGION
```

---

## Step 3 — Deploy with App Runner

```bash
aws apprunner create-service \
  --service-name mlops-backend \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "'$ECR'/mlops/backend:latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8000",
        "RuntimeEnvironmentVariables": {
          "MLOPS_SERVICE_URL": "http://mlops-service:8001"
        }
      }
    }
  }' \
  --instance-configuration '{"Cpu": "1 vCPU", "Memory": "2 GB"}' \
  --region $AWS_REGION
```

---

## Option B — Use AWS SageMaker for MLOps

```python
import boto3
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.model_monitor import DefaultModelMonitor

sm_client = boto3.client("sagemaker", region_name="eu-west-2")

def train_and_register(s3_data_uri: str, experiment_name: str) -> str:
    estimator = SKLearn(
        entry_point="train.py",
        source_dir="./mlops-service",
        instance_type="ml.m5.large",
        role=sagemaker.get_execution_role(),
        framework_version="1.2-1",
        hyperparameters={"max_iter": 1000}
    )
    estimator.fit({"training": s3_data_uri}, experiment_config={"ExperimentName": experiment_name})
    return estimator.latest_training_job.name

def setup_drift_monitor(endpoint_name: str, s3_baseline_uri: str) -> None:
    monitor = DefaultModelMonitor(role=sagemaker.get_execution_role(), instance_type="ml.m5.large")
    monitor.suggest_baseline(baseline_dataset=s3_baseline_uri, dataset_format={"csv": {"header": True}})
    monitor.create_monitoring_schedule(endpoint_input=endpoint_name, schedule_cron_expression="cron(0 * ? * * *)")
```

---

## Estimated Monthly Cost

| Service                    | Tier              | Est. Cost          |
|----------------------------|-------------------|--------------------|
| App Runner (backend)       | 1 vCPU / 2 GB     | ~$20–25/month      |
| App Runner (mlops-service) | 1 vCPU / 2 GB     | ~$20–25/month      |
| ECR + S3 + CloudFront      | Standard          | ~$3–7/month        |
| SageMaker (ml.m5.large)    | Per training hour | ~$0.115/hour       |
| SageMaker Model Monitor    | Per monitoring job| ~$0.20/hour        |
| **Total (Option A)**       |                   | **~$43–57/month**  |
| **Total (Option B)**       |                   | **~$23–32/month + training cost** |

For exact estimates → https://calculator.aws

---

## Teardown

```bash
aws ecr delete-repository --repository-name mlops/backend --force
aws ecr delete-repository --repository-name mlops/mlops-service --force
aws s3 rm s3://mlops-artifacts-$AWS_ACCOUNT --recursive
aws s3 rb s3://mlops-artifacts-$AWS_ACCOUNT
```
