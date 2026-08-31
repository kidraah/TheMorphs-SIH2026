from fastapi import APIRouter
from app.services.risk_service import risk_service

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

@router.get("")
def get_alerts(time: str = "2h"):
    """
    Returns active actionable alerts dynamically driven by the VARUNA model.
    """
    return risk_service.get_dynamic_alerts(time)

