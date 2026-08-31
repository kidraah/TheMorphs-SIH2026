from fastapi import APIRouter

router = APIRouter(prefix="/api")

@router.get("/health")
def health_check():
    """
    Basic health check endpoint for the backend.
    """
    return {"status": "ok", "service": "VARUNA Backend"}

@router.get("/data-sources/status")
def get_data_sources_status():
    """
    Returns telemetry for data sources.
    """
    return [
        {
            "id": "insat",
            "name": "INSAT-3D/3DR Satellite",
            "status": "Live",
            "type": "live",
            "frequency": "Every 15 min",
            "lastSync": "2 min ago"
        }
    ]

@router.get("/system/performance")
def get_system_performance():
    return [
        { "name": "Model Confidence", "percentage": 92, "status": "optimal" }
    ]
