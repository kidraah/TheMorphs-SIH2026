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
    overall_prob = np.maximum.reduce([preds["cloudburst"], preds["thunderstorm"], preds["flashFlood"]])
    precip = preds["cloudburst"]
    
    # Offload to background
    background_tasks.add_task(geospatial_service.process_background, overall_prob, precip)
    
    return ProcessResponse(status="Processing", message="Geospatial batch job queued in background")

@router.get("/heatmap")
async def get_heatmap():
    """
    Returns the generated PNG overlay as a base64 string with bounds.
    """
    if not geospatial_service.cached_heatmap:
        # Fallback trigger if not generated yet
        preds = risk_service._run_inference()
        import numpy as np
        overall_prob = np.maximum.reduce([preds["cloudburst"], preds["thunderstorm"], preds["flashFlood"]])
        geospatial_service.cached_heatmap = geospatial_service.generate_risk_heatmap(overall_prob)
        
    return geospatial_service.cached_heatmap

@router.get("/trajectories")
async def get_trajectories():
    """
    Retrieves the GeoJSON FeatureCollection of flood paths.
    """
    if not geospatial_service.cached_trajectories:
        preds = risk_service._run_inference()
        geospatial_service.cached_trajectories = geospatial_service.derive_flood_trajectories(preds["cloudburst"])
        
    return geospatial_service.cached_trajectories
