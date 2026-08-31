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
        # Bounding box exactly covering Uttarakhand & Himachal Pradesh combined
        # format: (west, south, east, north)
        self.bounds = (77.0, 29.5, 81.0, 31.5)
        self.cached_heatmap = None
        self.cached_trajectories = None
        
    def process_background(self, probability_array, precip_array):
        """Run the heavy spatial processing in the background and cache results"""
        self.cached_heatmap = self.generate_risk_heatmap(probability_array)
        self.cached_trajectories = self.derive_flood_polygons(precip_array)
        
    def generate_risk_heatmap(self, probability_array, colormap='inferno'):
        """
        Converts a 2D numpy probability array into a georeferenced PNG heatmap overlay.
        Masks out areas with very low probability to render them transparent.
        """
        # Apply a mild gaussian filter to smooth the edges without destroying peaks
        smoothed = gaussian_filter(probability_array, sigma=0.5)
        
        # Mask out values below 15% probability
        masked_array = np.ma.masked_where(smoothed < 0.15, smoothed)
        
        fig, ax = plt.subplots(figsize=(8, 8), frameon=False)
        
        # Ensure masked values are fully transparent
        cmap = matplotlib.colormaps.get_cmap(colormap).copy()
        cmap.set_bad(color='white', alpha=0)
        
        # Use a dynamic vmax so the highest probability always shines bright, but capped at 0.5 min
        dynamic_vmax = max(0.5, float(masked_array.max() if masked_array.count() > 0 else 1.0))
        
        ax.imshow(masked_array, cmap=cmap, vmin=0, vmax=dynamic_vmax, alpha=0.75)
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

    def derive_flood_polygons(self, precipitation_array):
        """
        Derive time-series expanding flood polygons from the probability array.
        Returns a list of GeoJSON FeatureCollections (one per frame).
        """
        h, w = precipitation_array.shape
        flow_acc = self.simulate_d8_flow(precipitation_array)
        
        threshold = np.percentile(flow_acc, 99.5) 
        y_indices, x_indices = np.where(flow_acc > threshold)
        
        west, south, east, north = self.bounds
        lon_step = (east - west) / w
        lat_step = (north - south) / h
        
        frames = []
        num_frames = 15
        
        # Select top distinct points
        points = []
        for i in range(min(20, len(y_indices))):
            points.append((x_indices[i], y_indices[i]))
            
        import math
        def get_circle_polygon(cx, cy, radius_deg, num_segments=16):
            coords = []
            for j in range(num_segments):
                angle = 2 * math.pi * j / num_segments
                coords.append([cx + math.cos(angle) * radius_deg, cy + math.sin(angle) * radius_deg])
            coords.append(coords[0]) # close polygon
            return coords

        for f in range(num_frames):
            features = []
            for (x, y) in points:
                lon = west + x * lon_step
                lat = north - y * lat_step
                
                # Expand radius over frames
                radius = 0.01 + (f * 0.003) 
                
                poly_coords = get_circle_polygon(lon, lat, radius)
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [poly_coords]
                    },
                    "properties": {
                        "frame": f,
                        "risk_level": "Severe"
                    }
                })
            
            frames.append({
                "type": "FeatureCollection",
                "features": features
            })
            
        return frames

geospatial_service = GeospatialService()
