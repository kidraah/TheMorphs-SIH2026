from fastapi import APIRouter

router = APIRouter(prefix="/api/weather", tags=["weather"])

@router.get("/nowcast")
def get_weather_nowcast():
    """
    Returns general meteorological information.
    """
    return {"status": "Processing"}
