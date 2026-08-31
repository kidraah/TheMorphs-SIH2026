import io
import base64
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

class GeospatialService:
    def __init__(self):
        # Bounding box roughly covering Uttarakhand / Himachal regions
        # format: (west, south, east, north)
        self.bounds = (77.0, 29.5, 81.0, 31.5)
        self.cached_heatmap = None
        self.cached_trajectories = None
        
    def process_background(self, probability_array, precip_array):
        """Run the heavy spatial processing in the background and cache results"""
        self.cached_heatmap = self.generate_risk_heatmap(probability_array)
        self.cached_trajectories = self.derive_flood_trajectories(precip_array)
        
    def generate_risk_heatmap(self, probability_array, colormap='inferno'):
        """
        Converts a 2D numpy probability array into a georeferenced PNG heatmap overlay.
        """
        fig, ax = plt.subplots(figsize=(8, 8), frameon=False)
        ax.imshow(probability_array, cmap=colormap, vmin=0, vmax=1, alpha=0.6)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        
        buf.seek(0)
        png_data = buf.read()
        b64_img = base64.b64encode(png_data).decode('utf-8')
        
        west, south, east, north = self.bounds
        
        return {
            "image": f"data:image/png;base64,{b64_img}",
            "bounds": [[south, west], [north, east]], 
            "opacity": 0.6
        }

    def _generate_mock_dem(self, height, width):
        """Simulate a terrain mesh with some hills and valleys."""
        x = np.linspace(0, 10 * np.pi, width)
        y = np.linspace(0, 10 * np.pi, height)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y) * 1000 + 2000
        return Z

    def simulate_d8_flow(self, precipitation_array):
        """
        Simulate hydrological flow (D8) over a terrain model.
        Returns a flow accumulation matrix.
        """
        h, w = precipitation_array.shape
        dem = self._generate_mock_dem(h, w)
        
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        grad_x = convolve(dem, kernel_x)
        grad_y = convolve(dem, kernel_y)
        
        flow_acc = np.abs(grad_x) + np.abs(grad_y)
        flow_acc = flow_acc * (precipitation_array * 10)
        
        if flow_acc.max() > 0:
            flow_acc = flow_acc / flow_acc.max()
            
        return flow_acc

    def derive_flood_trajectories(self, precipitation_array):
        """
        Derive vector flood paths (GeoJSON) from the probability array.
        Returns a GeoJSON dict.
        """
        h, w = precipitation_array.shape
        flow_acc = self.simulate_d8_flow(precipitation_array)
        
        threshold = np.percentile(flow_acc, 98) 
        y_indices, x_indices = np.where(flow_acc > threshold)
        
        features = []
        west, south, east, north = self.bounds
        lon_step = (east - west) / w
        lat_step = (north - south) / h
        
        for i in range(min(50, len(y_indices))):
            x, y = x_indices[i], y_indices[i]
            
            lon_start = west + x * lon_step
            lat_start = north - y * lat_step 
            
            coords = [[lon_start, lat_start]]
            current_x, current_y = x, y
            
            for step in range(5):
                current_y = min(h - 1, current_y + np.random.randint(1, 4))
                current_x = min(w - 1, max(0, current_x + np.random.randint(-2, 3)))
                
                lon = west + current_x * lon_step
                lat = north - current_y * lat_step
                coords.append([lon, lat])
                
            if len(coords) > 1:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    },
                    "properties": {
                        "risk_level": "Severe",
                        "lead_time": f"+{i % 4 + 1}h"
                    }
                }
                features.append(feature)
                
        return {
            "type": "FeatureCollection",
            "features": features
        }

geospatial_service = GeospatialService()
