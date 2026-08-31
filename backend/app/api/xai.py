from fastapi import APIRouter
from app.services.risk_service import risk_service

router = APIRouter(prefix="/api/xai", tags=["xai"])

@router.get("/triggers")
def get_xai_triggers():
    """
    Returns explainable AI insights based on model outputs.
    """
    return risk_service.get_dynamic_xai()

