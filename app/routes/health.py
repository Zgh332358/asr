"""Health probes for cloud-backed transcription (no local GPU required)."""
from fastapi import APIRouter, HTTPException, Request
from app.config import settings
from app.schemas.responses import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])

@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok")

@router.get("/health/ready", response_model=ReadyResponse)
async def readiness(request: Request) -> ReadyResponse:
    worker = getattr(request.app.state, "worker", None)
    if worker is None or not worker.is_ready:
        raise HTTPException(status_code=503, detail={
            "status": "not_ready", "provider": "stepfun",
            "configured": settings.cloud_configured,
            "message": "Set OPENAI_API_KEY and restart the service.",
        })
    return ReadyResponse(status="ready", model_loaded=True, gpu_available=False,
                         configured=True, model=settings.asr_model)

@router.get("/health/gpu")
async def gpu_info():
    raise HTTPException(status_code=404, detail="Cloud speech does not use a local GPU.")
