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
        },
        {
            "id": "imdaa",
            "name": "IMDAA High-Res Reanalysis",
            "status": "Live",
            "type": "live",
            "frequency": "Every 6 hours",
            "lastSync": "1 hr ago"
        },
        {
            "id": "dwr",
            "name": "Doppler Weather Radar (DWR)",
            "status": "Live",
            "type": "live",
            "frequency": "Every 10 min",
            "lastSync": "4 min ago"
        },
        {
            "id": "arg",
            "name": "Automated Rain Gauge (ARG)",
            "status": "Live",
            "type": "live",
            "frequency": "Hourly",
            "lastSync": "12 min ago"
        },
        {
            "id": "dem",
            "name": "SRTM Digital Elevation Model",
            "status": "Static",
            "type": "static",
            "frequency": "Baseline",
            "lastSync": "N/A"
        }
    ]

@router.get("/system/performance")
def get_system_performance():
    return [
        { "name": "Model Confidence", "percentage": 92, "status": "optimal" }
    ]
