from fastapi import APIRouter

from app.data_store import data_store

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Liveness/readiness check")
def health_check():
    return {
        "status": "ok",
        "service": "MPLADS Analytics & Monitoring API",
        "data_loaded": data_store._loaded,
    }
