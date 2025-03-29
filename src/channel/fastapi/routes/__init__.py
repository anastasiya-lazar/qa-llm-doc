from fastapi import APIRouter, Request
from .documents import router as documents_router
from .questions import router as questions_router
from src.channel.fastapi.config import get_settings
from .base import limiter

settings = get_settings()

# Create the main router
router = APIRouter()


# Health check endpoint
@router.get("/")
@limiter.limit("5/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    return {"status": "ok", "message": settings.PROJECT_NAME}


# Include all sub-routers
router.include_router(documents_router)
router.include_router(questions_router)

__all__ = ["router"]
