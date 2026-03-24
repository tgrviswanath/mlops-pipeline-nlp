from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..core.service import train_model, predict, get_experiments, get_model_versions, get_metrics, check_drift

router = APIRouter(prefix="/api/v1/mlops")

class TrainRequest(BaseModel):
    params: dict = {"C": 1.0, "max_iter": 100}

class PredictRequest(BaseModel):
    text: str

@router.post("/train")
async def train(req: TrainRequest):
    try:
        return train_model(req.params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict_endpoint(req: PredictRequest):
    try:
        return predict(req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/experiments")
async def experiments():
    return get_experiments()

@router.get("/versions")
async def versions():
    return get_model_versions()

@router.get("/metrics")
async def metrics():
    return get_metrics()

@router.get("/drift")
async def drift():
    return check_drift()
