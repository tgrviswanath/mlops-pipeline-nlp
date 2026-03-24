from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..core.service import call
import httpx

router = APIRouter(prefix="/api/v1")

def _handle(e):
    if isinstance(e, httpx.HTTPStatusError):
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    raise HTTPException(status_code=500, detail=str(e))

class TrainRequest(BaseModel):
    params: dict = {"C": 1.0, "max_iter": 100}

class PredictRequest(BaseModel):
    text: str

@router.post("/train")
async def train(req: TrainRequest):
    try:
        return await call("post", "/train", json=req.model_dump())
    except Exception as e:
        _handle(e)

@router.post("/predict")
async def predict(req: PredictRequest):
    try:
        return await call("post", "/predict", json=req.model_dump())
    except Exception as e:
        _handle(e)

@router.get("/experiments")
async def experiments():
    try:
        return await call("get", "/experiments")
    except Exception as e:
        _handle(e)

@router.get("/versions")
async def versions():
    try:
        return await call("get", "/versions")
    except Exception as e:
        _handle(e)

@router.get("/metrics")
async def metrics():
    try:
        return await call("get", "/metrics")
    except Exception as e:
        _handle(e)

@router.get("/drift")
async def drift():
    try:
        return await call("get", "/drift")
    except Exception as e:
        _handle(e)
