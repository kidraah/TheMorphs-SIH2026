from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import asyncio
from app.services.geospatial_service import geospatial_service
from app.services.risk_service import risk_service

router = APIRouter(prefix="/api/geo", tags=["Geospatial"])

class ProcessResponse(BaseModel):
    status: str
    message: str

@router.post("/process", response_model=ProcessResponse)
async def trigger_geospatial_processing(background_tasks: BackgroundTasks):
    """
    Manually trigger the asynchronous batch processing pipeline.
    """
    # Fetch latest probabilities from model
    preds = risk_service._run_inference()
    
    # We'll use cloudburst for precipitation simulation, and max of all for overall heatmap
    import numpy as np
    overall_prob = np.squeeze(np.maximum.reduce([preds["cloudburst"], preds["thunderstorm"], preds["flashFlood"]]))
    precip = np.squeeze(preds["cloudburst"])
    
    # Offload to background
    background_tasks.add_task(geospatial_service.process_background, overall_prob, precip)
    
    return ProcessResponse(status="Processing", message="Geospatial batch job queued in background")

@router.get("/heatmap")
def get_heatmap(time: str = "2h"):
    """
    Returns the generated PNG overlay as a base64 string with bounds.
    """
    if not isinstance(geospatial_service.cached_heatmap, dict):
        geospatial_service.cached_heatmap = {}
        
    if time not in geospatial_service.cached_heatmap:
        scale = 1.0
        if time == "now": scale = 0.4
        elif time == "2h": scale = 0.7
        elif time == "4h": scale = 1.0
        elif time == "6h": scale = 0.5
        
        preds = risk_service._run_inference()
        import numpy as np
        overall_prob = np.squeeze(np.maximum.reduce([preds["cloudburst"], preds["thunderstorm"], preds["flashFlood"]])) * scale
        geospatial_service.cached_heatmap[time] = geospatial_service.generate_risk_heatmap(overall_prob)
        
    return geospatial_service.cached_heatmap[time]

@router.get("/radar")
def get_radar_nowcast(time: str = "2h"):
    """
    Returns a radar-styled precipitation heatmap base64 image.
    """
    preds = risk_service._run_inference()
    
    scale = 1.0
    if time == "now": scale = 0.4
    elif time == "2h": scale = 0.7
    elif time == "4h": scale = 1.0
    elif time == "6h": scale = 0.5
    
    # Use cloudburst tensor and apply efficiency scaler to match others
    import numpy as np
    cb_prob = np.squeeze(preds["cloudburst"]) * 0.75 * scale
    
    return geospatial_service.generate_radar_image(cb_prob)

@router.get("/trajectories")
def get_trajectories():
    """
    Retrieves the GeoJSON FeatureCollection of flood paths.
    """
    if not geospatial_service.cached_trajectories:
        preds = risk_service._run_inference()
        geospatial_service.cached_trajectories = geospatial_service.derive_flood_trajectories(preds["cloudburst"])
        
    return geospatial_service.cached_trajectories
