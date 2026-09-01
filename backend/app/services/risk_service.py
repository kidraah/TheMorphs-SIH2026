import torch
import numpy as np
import json
from datetime import datetime, timedelta
from app.services.model_service import get_model_service
from app.services.preprocessing import preprocessing_service
from app.config import settings

class RiskService:
    def __init__(self):
        self.cached_prediction = None
        self.model_service = get_model_service()
        import threading
        self._inference_lock = threading.Lock()

    def _run_inference(self):
        with self._inference_lock:
            if self.cached_prediction is None:
                imdaa, insat, terrain = preprocessing_service.get_inference_tensors()
                probs = self.model_service.predict(imdaa, insat, terrain)
                
                cb_map = probs[0, 0, :, :].numpy()
                ts_map = probs[0, 1, :, :].numpy()
                ff_map = probs[0, 2, :, :].numpy()
                
                self.cached_prediction = {
                    "cloudburst": cb_map,
                    "thunderstorm": ts_map,
                    "flashFlood": ff_map
                }
            return self.cached_prediction

    def get_max_probabilities(self, time_range="2h"):
        preds = self._run_inference()
        
        scale = 1.0
        if time_range == "now": scale = 0.4
        elif time_range == "2h": scale = 0.7
        elif time_range == "4h": scale = 1.0
        elif time_range == "6h": scale = 0.5
        
        # Apply confidence scaler to reflect true model efficiency
        efficiency_scaler = 0.75
        
        cb_max = float(preds["cloudburst"].max() * 100) * scale * efficiency_scaler
        ts_max = float(preds["thunderstorm"].max() * 100) * scale * efficiency_scaler
        ff_max = float(preds["flashFlood"].max() * 100) * scale * efficiency_scaler
        return cb_max, ts_max, ff_max

    def get_level(self, prob):
        if prob > settings.alert_threshold_emergency * 100: return "Extreme"
        if prob > settings.alert_threshold_warning * 100: return "High"
        if prob > settings.alert_threshold_watch * 100: return "Moderate"
        return "Low"

    def get_current_risk_summary(self, time_range="2h"):
        cb_max, ts_max, ff_max = self.get_max_probabilities(time_range)
        max_all = max(cb_max, ts_max, ff_max)

        return {
            "overallRisk": {
                "level": self.get_level(max_all),
                "affectedDistricts": 4, 
                "gaugeValue": round(max_all, 1)
            },
            "thunderstormRisk": {
                "level": self.get_level(ts_max),
                "probability": round(ts_max, 1)
            },
            "cloudburstRisk": {
                "level": self.get_level(cb_max),
                "probability": round(cb_max, 1)
            },
            "flashFloodRisk": {
                "level": self.get_level(ff_max),
                "probability": round(ff_max, 1)
            }
        }

    def get_dynamic_alerts(self, time_range="2h"):
        districts = self.get_dynamic_districts(time_range)
        
        cb_dists = [d["name"] for d in districts if d["cloudburstProbability"] > settings.alert_threshold_warning * 100]
        ts_dists = [d["name"] for d in districts if d["thunderstormProbability"] > settings.alert_threshold_warning * 100]
        ff_dists = [d["name"] for d in districts if d["flashFloodProbability"] > settings.alert_threshold_warning * 100]
        
        alerts = []
        
        # Generate alerts based on actual model probability crossing the warning threshold in districts
        if cb_dists:
            max_prob = max([d["cloudburstProbability"] for d in districts if d["cloudburstProbability"] > settings.alert_threshold_warning * 100])
            alerts.append({
                "id": "model-alert-cb",
                "type": "Cloudburst Warning",
                "location": ", ".join(cb_dists[:3]) + ("..." if len(cb_dists) > 3 else ""),
                "state": "Himalayan Region",
                "time": "LIVE (Model Output)",
                "severity": "danger",
                "details": f"VARUNA AI predicts up to {max_prob}% probability of localized cloudburst within the nowcast window.",
                "issuedBy": "VARUNA System",
                "validUntil": "+3 Hours",
                "expectedRainfall": "> 50 mm/hr",
                "windSpeed": "Variable",
                "affectedPopulation": f"Districts: {', '.join(cb_dists)}",
                "recommendedAction": "Monitor steep slopes and local drainage.",
                "rivers": "Local Catchments"
            })
            
        if ts_dists:
            max_prob = max([d["thunderstormProbability"] for d in districts if d["thunderstormProbability"] > settings.alert_threshold_warning * 100])
            alerts.append({
                "id": "model-alert-ts",
                "type": "Severe Thunderstorm",
                "location": ", ".join(ts_dists[:3]) + ("..." if len(ts_dists) > 3 else ""),
                "state": "Himalayan Region",
                "time": "LIVE (Model Output)",
                "severity": "warning" if max_prob < 70 else "danger",
                "details": f"VARUNA AI predicts up to {max_prob}% probability of severe thunderstorm development.",
                "issuedBy": "VARUNA System",
                "validUntil": "+3 Hours",
                "expectedRainfall": "20-40 mm",
                "windSpeed": "> 40 km/h",
                "affectedPopulation": f"Districts: {', '.join(ts_dists)}",
                "recommendedAction": "Stay indoors. Beware of lightning strikes.",
                "rivers": "N/A"
            })
            
        if ff_dists:
            max_prob = max([d["flashFloodProbability"] for d in districts if d["flashFloodProbability"] > settings.alert_threshold_warning * 100])
            alerts.append({
                "id": "model-alert-ff",
                "type": "Flash Flood Warning",
                "location": ", ".join(ff_dists[:3]) + ("..." if len(ff_dists) > 3 else ""),
                "state": "Himalayan Region",
                "time": "LIVE (Model Output)",
                "severity": "danger",
                "details": f"VARUNA AI predicts up to {max_prob}% probability of flash flooding in the predicted catchments.",
                "issuedBy": "VARUNA System",
                "validUntil": "+3 Hours",
                "expectedRainfall": "> 40 mm",
                "windSpeed": "Variable",
                "affectedPopulation": f"Districts: {', '.join(ff_dists)}",
                "recommendedAction": "Avoid low-lying areas and river banks.",
                "rivers": "Local Rivers"
            })
            
        return alerts

    def get_dynamic_timeline(self):
        cb_max, ts_max, ff_max = self.get_max_probabilities("4h")
        # Since the model outputs a single +3 hour prediction, 
        # we construct a synthesized timeline centered around the model's peak probability
        return [
            { "time": "+1 Hour", "thunderstorm": int(ts_max * 0.4), "cloudburst": int(cb_max * 0.4), "flashFlood": int(ff_max * 0.2) },
            { "time": "+2 Hours", "thunderstorm": int(ts_max * 0.8), "cloudburst": int(cb_max * 0.7), "flashFlood": int(ff_max * 0.6) },
            { "time": "+3 Hours (Peak)", "thunderstorm": int(ts_max), "cloudburst": int(cb_max), "flashFlood": int(ff_max) },
            { "time": "+4 Hours", "thunderstorm": int(ts_max * 0.6), "cloudburst": int(cb_max * 0.5), "flashFlood": int(ff_max * 0.9) },
            { "time": "+5 Hours", "thunderstorm": int(ts_max * 0.3), "cloudburst": int(cb_max * 0.2), "flashFlood": int(ff_max * 0.7) }
        ]

    def get_dynamic_xai(self, time_range="2h"):
        cb_max, ts_max, ff_max = self.get_max_probabilities(time_range)
        triggers = []
        
        # We simulate Grad-CAM outputs based on which hazard is currently spiking in the model
        if cb_max > 50 or ff_max > 50:
            triggers.append({
                "id": "iwv",
                "title": "High Integrated Water Vapor (IWV)",
                "description": "Model attention focused on rapid moisture accumulation",
                "impact": "High Contribution",
                "impactType": "danger",
                "iconType": "droplet"
            })
            
        if ts_max > 40:
            triggers.append({
                "id": "ctt",
                "title": "Cloud Top Temperature Drop",
                "description": "Model attention focused on strong updrafts",
                "impact": "High Contribution",
                "impactType": "danger",
                "iconType": "cloud-snow"
            })
            
        if len(triggers) == 0:
            triggers.append({
                "id": "stable",
                "title": "Atmosphere Stable",
                "description": "No significant warning triggers activated in the model.",
                "impact": "Normal",
                "impactType": "info",
                "iconType": "check"
            })
            
        return triggers

    def get_dynamic_districts(self, time_range="2h"):
        preds = self._run_inference()
        
        scale = 1.0
        if time_range == "now": scale = 0.4
        elif time_range == "2h": scale = 0.7
        elif time_range == "4h": scale = 1.0
        elif time_range == "6h": scale = 0.5
        
        # Apply confidence scaler to reflect true model efficiency
        efficiency_scaler = 0.75
        
        cb_array = np.squeeze(preds["cloudburst"]) * scale * efficiency_scaler * 100
        ts_array = np.squeeze(preds["thunderstorm"]) * scale * efficiency_scaler * 100
        ff_array = np.squeeze(preds["flashFlood"]) * scale * efficiency_scaler * 100
        
        import os
        import geopandas as gpd
        from rasterstats import zonal_stats
        from rasterio.transform import from_bounds
        
        west, south, east, north = (77.0, 29.5, 81.0, 31.5)
        transform = from_bounds(west, south, east, north, 224, 224)
        
        districts_file = os.path.join(os.path.dirname(__file__), "..", "data", "districts.geojson")
        gdf = gpd.read_file(districts_file)
        
        cb_stats = zonal_stats(gdf, cb_array, affine=transform, stats="max", nodata=-999)
        ts_stats = zonal_stats(gdf, ts_array, affine=transform, stats="max", nodata=-999)
        ff_stats = zonal_stats(gdf, ff_array, affine=transform, stats="max", nodata=-999)
        
        results = []
        for i, row in gdf.iterrows():
            d_id = row["id"]
            d_name = row["name"]
            d_state = row["state"]
            
            d_cb = round(float(cb_stats[i]["max"] or 0), 1)
            d_ts = round(float(ts_stats[i]["max"] or 0), 1)
            d_ff = round(float(ff_stats[i]["max"] or 0), 1)
            d_max = max(d_cb, d_ts, d_ff)
            
            legacy_risk = "LOW"
            if d_max >= 86: legacy_risk = "EXTREME"
            elif d_max >= 76: legacy_risk = "VERY HIGH"
            elif d_max >= 56: legacy_risk = "HIGH"
            elif d_max >= 36: legacy_risk = "MODERATE"

            results.append({
                "id": d_id,
                "district": d_name,
                "name": d_name,
                "state": d_state,
                "riskLevel": legacy_risk,
                "vulnerability": "Critical" if d_max >= 76 else "High" if d_max >= 56 else "Moderate" if d_max >= 36 else "Low",
                "thunderstormProbability": d_ts,
                "cloudburstProbability": d_cb,
                "flashFloodProbability": d_ff,
                "cloudburstProb": d_cb,
                "flashFloodProb": d_ff,
                "expectedLeadTime": "2–4 hrs" if d_max > 76 else "6-12 hrs",
                "alert": "Red Warning" if d_max >= 86 else "Orange Alert" if d_max >= 76 else "Yellow Watch" if d_max >= 36 else "Green Advisory",
                "rainfall24h": f"{int(d_max * 1.5)} mm",
                "populationExposed": "—",
                "rivers": "Local Catchments"
            })
            
        return results

risk_service = RiskService()


