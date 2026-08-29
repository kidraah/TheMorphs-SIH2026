"""
Script 3: Bhoonidhi CartoDEM -- Topographic Dynamics Analysis
SIH26077 -- Exploratory Data Analysis & Precursor Extraction

Computes terrain derivatives for flash flood vulnerability assessment:
  1. DEM Elevation Map -- baseline terrain visualization
  2. Slope Map -- steep terrain identification (>30 deg = landslide/channel risk)
  3. Aspect Map -- drainage direction analysis (compass bearing)

Data:  CartoDEM 30m resolution GeoTIFF from dataset_root/dem/cartodem.tif
Output: PNG saved to output/dem_topography.png

NOTE: The full DEM is ~3.1 GB. This script uses rasterio windowed reading
to load only a sub-region (North India Himalayas) to avoid OOM.
"""

import matplotlib
matplotlib.use('Agg')

import rasterio
from rasterio.windows import from_bounds
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DEM_PATH = "C:/THEMORPHS/dataset_root/dem/cartodem.tif"
OUTPUT_DIR = "C:/THEMORPHS/output"

# Sub-region bounding box (Uttarakhand / Himachal -- flash flood prone)
BBOX = {
    'west':  76.0,
    'east':  81.0,
    'south': 29.0,
    'north': 33.0
}

STEEP_THRESHOLD_DEG = 30.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Core Functions
# ============================================================

def load_dem_windowed(dem_path, bbox, max_pixels=3000):
    """Load a sub-region of the DEM using rasterio windowed reading."""
    with rasterio.open(dem_path) as src:
        print(f"   Full DEM: {src.width}x{src.height} pixels, CRS={src.crs}")
        print(f"   Bounds: {src.bounds}")

        window = from_bounds(
            bbox['west'], bbox['south'], bbox['east'], bbox['north'],
            src.transform
        )
        window = window.round_offsets().round_lengths()

        win_height = int(window.height)
        win_width = int(window.width)
        print(f"   Window: {win_width}x{win_height} pixels")

        factor = max(1, max(win_width, win_height) // max_pixels)
        out_height = win_height // factor
        out_width = win_width // factor

        if factor > 1:
            print(f"   Downsampling by {factor}x -> {out_width}x{out_height}")

        elevation = src.read(
            1,
            window=window,
            out_shape=(out_height, out_width)
        ).astype(np.float32)

        win_transform = src.window_transform(window)

        if factor > 1:
            win_transform = rasterio.transform.Affine(
                win_transform.a * factor,
                win_transform.b,
                win_transform.c,
                win_transform.d,
                win_transform.e * factor,
                win_transform.f
            )

        pixel_size_x_deg = abs(win_transform.a)
        pixel_size_y_deg = abs(win_transform.e)
        ref_lat_rad = np.deg2rad((bbox['north'] + bbox['south']) / 2.0)
        dx_m = pixel_size_x_deg * 111320.0 * np.cos(ref_lat_rad)
        dy_m = pixel_size_y_deg * 110540.0

    return elevation, win_transform, dx_m, dy_m


def compute_slope_aspect(elevation, dx_m, dy_m):
    """Compute terrain slope (degrees) and aspect (0-360 compass bearing)."""
    dz_dy, dz_dx = np.gradient(elevation, dy_m, dx_m)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (aspect_deg + 360) % 360

    # Mask flat areas from aspect (direction is meaningless there)
    aspect_deg = np.ma.masked_where(slope_deg < 0.5, aspect_deg)

    return slope_deg, aspect_deg


def build_extent(transform, width, height):
    """Build [left, right, bottom, top] extent for imshow."""
    left = transform.c
    right = transform.c + width * transform.a
    top = transform.f
    bottom = transform.f + height * transform.e
    return [left, right, bottom, top]


# ============================================================
# Main Analysis
# ============================================================

print("=" * 65)
print("  SCRIPT 3: CARTODEM TOPOGRAPHIC DYNAMICS ANALYSIS")
print("  SIH26077 -- Flash Flood Vulnerability Assessment")
print("=" * 65)
print(f"\n  Region: {BBOX['south']}-{BBOX['north']}N, {BBOX['west']}-{BBOX['east']}E")
print(f"  Source: {DEM_PATH}")


# ----------------------------------------------------------
# Load DEM
# ----------------------------------------------------------
print("\n[1/3] Loading DEM (windowed read)...")

if not os.path.exists(DEM_PATH):
    print(f"  ERROR: DEM file not found: {DEM_PATH}")
    exit(1)

elevation, transform, dx_m, dy_m = load_dem_windowed(DEM_PATH, BBOX, max_pixels=3000)

# Mask no-data / ocean
elevation = np.ma.masked_where((elevation < -100) | (elevation > 9000), elevation)

extent = build_extent(transform, elevation.shape[1], elevation.shape[0])

print(f"   Elevation range: {float(np.nanmin(elevation)):.0f} -- {float(np.nanmax(elevation)):.0f} m")
print(f"   Pixel size: ~{dx_m:.1f}m x {dy_m:.1f}m")


# ----------------------------------------------------------
# Compute Slope & Aspect
# ----------------------------------------------------------
print("\n[2/3] Computing slope and aspect...")

slope_deg, aspect_deg = compute_slope_aspect(elevation.filled(np.nan), dx_m, dy_m)

if hasattr(elevation, 'mask'):
    slope_deg = np.ma.masked_where(elevation.mask, slope_deg)
    aspect_deg = np.ma.masked_where(elevation.mask, aspect_deg)

steep_mask = slope_deg > STEEP_THRESHOLD_DEG
steep_pct = 100 * np.sum(steep_mask) / slope_deg.size

print(f"   Slope range: {float(np.nanmin(slope_deg)):.1f} -- {float(np.nanmax(slope_deg)):.1f} deg")
print(f"   Mean slope: {float(np.nanmean(slope_deg)):.1f} deg")
print(f"   Steep terrain (>{STEEP_THRESHOLD_DEG} deg): {steep_pct:.1f}% of region")


# ----------------------------------------------------------
# Plot Dashboard
# ----------------------------------------------------------
print("\n[3/3] Rendering topography dashboard...")

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle(
    f'CartoDEM Topographic Analysis -- North India Himalayas\n'
    f'{BBOX["south"]}-{BBOX["north"]}N, {BBOX["west"]}-{BBOX["east"]}E | '
    f'SIH26077 Flash Flood Vulnerability',
    fontsize=15, fontweight='bold', y=1.02
)


# --- Panel 1: Elevation Map ---
ax1 = axes[0]
elev_valid = elevation.compressed() if hasattr(elevation, 'compressed') else elevation.ravel()
im1 = ax1.imshow(
    elevation, cmap='terrain', extent=extent, origin='upper',
    vmin=float(np.nanpercentile(elev_valid, 2)),
    vmax=float(np.nanpercentile(elev_valid, 98))
)
plt.colorbar(im1, ax=ax1, label='Elevation (m)', fraction=0.046, pad=0.04, shrink=0.85)
ax1.set_title('Digital Elevation Model (CartoDEM 30m)\nBase Terrain for Flood Routing', fontsize=11)
ax1.set_xlabel('Longitude (E)')
ax1.set_ylabel('Latitude (N)')


# --- Panel 2: Slope Map ---
ax2 = axes[1]
im2 = ax2.imshow(
    slope_deg, cmap='YlOrRd', extent=extent, origin='upper',
    vmin=0, vmax=60
)
plt.colorbar(im2, ax=ax2, label='Slope (degrees)', fraction=0.046, pad=0.04, shrink=0.85)

steep_overlay = np.ma.masked_where(~steep_mask, slope_deg)
ax2.contour(
    steep_overlay, levels=[STEEP_THRESHOLD_DEG], colors='darkred',
    linewidths=0.5, extent=extent, origin='upper'
)

ax2.set_title(
    f'Terrain Slope -- Steep Drainage Channels\n'
    f'Red zones (>{STEEP_THRESHOLD_DEG} deg) = Flash flood funneling risk ({steep_pct:.1f}%)',
    fontsize=11
)
ax2.set_xlabel('Longitude (E)')
ax2.set_ylabel('Latitude (N)')


# --- Panel 3: Aspect Map ---
ax3 = axes[2]
im3 = ax3.imshow(
    aspect_deg, cmap='hsv', extent=extent, origin='upper',
    vmin=0, vmax=360
)
cbar3 = plt.colorbar(im3, ax=ax3, label='Aspect (degrees)', fraction=0.046, pad=0.04, shrink=0.85)
cbar3.set_ticks([0, 90, 180, 270, 360])
cbar3.set_ticklabels(['N', 'E', 'S', 'W', 'N'])

ax3.set_title(
    'Terrain Aspect -- Drainage Flow Direction\n'
    'Indicates which direction steep slopes face',
    fontsize=11
)
ax3.set_xlabel('Longitude (E)')
ax3.set_ylabel('Latitude (N)')


# ============================================================
# Save Output
# ============================================================
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "dem_topography.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n" + "=" * 65)
print(f"  DONE -- Dashboard saved: {output_path}")
print("=" * 65)
print("\nKey Takeaways for Model Pipeline:")
print(f"  - {steep_pct:.1f}% of region has slope > {STEEP_THRESHOLD_DEG} deg -- natural flood channels")
print("  - Aspect reveals dominant drainage direction (S/SE facing = monsoon-exposed)")
print("  - DEM is a STATIC feature layer -- overlay with atmospheric probability maps")
print("  -> Slope + aspect form the flash flood head's terrain input features.\n")
