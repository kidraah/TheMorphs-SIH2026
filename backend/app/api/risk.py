from fastapi import APIRouter
from app.services.risk_service import risk_service

router = APIRouter(prefix="/api/risk", tags=["risk"])

@router.get("/current")
def get_current_risk(time: str = "2h"):
    """
    Returns the overall risk metrics and hazard-specific probabilities
    based on the latest VARUNA model inference.
    """
    return risk_service.get_current_risk_summary(time)

@router.get("/districts")
def get_district_risks(time: str = "2h"):
    """
    Returns district-level aggregated risks and probabilities
    dynamically generated from the AI model output.
    """
    return risk_service.get_dynamic_districts(time)


@router.get("/timeline")
def get_risk_timeline():
    """
    Returns 2-6 hour timeline predictions based on the AI model.
    """
    return risk_service.get_dynamic_timeline()
