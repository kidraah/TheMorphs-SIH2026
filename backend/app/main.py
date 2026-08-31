from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import status, risk, xai, alerts, weather, geospatial

app = FastAPI(
    title="VARUNA Backend",
    description="Backend for the AI-Driven Hyper-Local Early Warning System",
    version="1.0.0"
)

# Set up CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(status.router)
app.include_router(risk.router)
app.include_router(xai.router)
app.include_router(alerts.router)
app.include_router(weather.router)
app.include_router(geospatial.router)
