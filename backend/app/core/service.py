import httpx
from .config import settings

BASE = f"{settings.MLOPS_SERVICE_URL}/api/v1/mlops"

async def call(method, path, **kwargs):
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await getattr(client, method)(f"{BASE}{path}", **kwargs)
        r.raise_for_status()
        return r.json()
