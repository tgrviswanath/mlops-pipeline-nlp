import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.service import train_model, predict, get_experiments, get_model_versions, get_metrics, check_drift

router = APIRouter(prefix="/api/v1/mlops")


class TrainRequest(BaseModel):
    params: dict = {"C": 1.0, "max_iter": 100}


class PredictRequest(BaseModel):
    text: str


@router.post("/train")
async def train(req: TrainRequest):
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, train_model, req.params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_endpoint(req: PredictRequest):
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, predict, req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
def experiments():
    return get_experiments()


@router.get("/versions")
def versions():
    return get_model_versions()


@router.get("/metrics")
def metrics():
    return get_metrics()


@router.get("/drift")
def drift():
    return check_drift()
