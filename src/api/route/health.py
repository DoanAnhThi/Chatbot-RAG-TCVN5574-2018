from datetime import datetime
from fastapi import APIRouter

from src.api.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Service is running",
        timestamp=datetime.utcnow().isoformat()
    )
