"""
Script 2: IMDAA Reanalysis -- Thermodynamic & Kinematic Analysis
SIH26077 -- Exploratory Data Analysis & Precursor Extraction

Computes three atmospheric instability/dynamics fields from IMDAA data:
  1. Temperature Lapse Rate (T_850 - T_500) -- CAPE/instability proxy
  2. Deep-Layer Bulk Wind Shear (300-850 hPa) -- storm severity indicator
  3. Low-Level Convergence (925 hPa) -- lift trigger for storm initiation

Data:  IMDAA reanalysis .nc files from dataset_root/imdaa/
Output: PNG saved to output/imdaa_thermodynamics.png
"""

import matplotlib
matplotlib.use('Agg')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
ROOT = "C:/THEMORPHS/dataset_root/imdaa"
OUTPUT_DIR = "C:/THEMORPHS/output"
TIMESTAMP = "2019081600"  # Aug 16, 2019 00:00 UTC
HOUR = "00"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helper Functions
# ============================================================

def build_imdaa_path(var_prefix, level_mb, timestamp, hour):
    """Construct IMDAA file path from variable, level, and timestamp."""
    filename = f"{var_prefix}-{level_mb}mb_{timestamp}_ncum_imdaa_reanl_prl_{hour}_{level_mb}_hpa.nc"
    return os.path.join(ROOT, filename)


def load_imdaa_field(var_prefix, level_mb, timestamp, hour):
    """Load a single IMDAA field, returning 2D data + coordinates."""
    path = build_imdaa_path(var_prefix, level_mb, timestamp, hour)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

    ds = xr.open_dataset(path)
    var_name = list(ds.data_vars)[0]
    data = ds[var_name].squeeze().values
    lat = ds['lat'].values
    lon = ds['lon'].values
    ds.close()

    return data, lat, lon, var_name


def add_india_map_features(ax):
    """Add geographic features to a cartopy axis focused on India."""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#333')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', color='#555')
    ax.add_feature(
        cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m'),
        linewidth=0.3, edgecolor='#888', facecolor='none'
    )
    ax.set_extent([65, 100, 5, 40], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False


# ============================================================
# Main Analysis
# ============================================================

print("=" * 65)
print("  SCRIPT 2: IMDAA THERMODYNAMIC & KINEMATIC ANALYSIS")
print("  SIH26077 -- Severe Weather Precursor Extraction")
print("=" * 65)
print(f"\n  Timestamp: {TIMESTAMP} ({HOUR} UTC)")
print(f"  Source:    {ROOT}")

fig = plt.figure(figsize=(22, 8))
gs = fig.add_gridspec(1, 3, wspace=0.28)
fig.suptitle(
    f'IMDAA Reanalysis -- Atmospheric Instability & Dynamics\n'
    f'Aug 16, 2019 00:00 UTC | SIH26077 Precursor Analysis',
    fontsize=16, fontweight='bold', y=1.02
)


# ----------------------------------------------------------
# Panel 1: Temperature Lapse Rate (T_850 - T_500)
# ----------------------------------------------------------
print("\n[1/3] Computing Temperature Lapse Rate (T_850 - T_500)...")

try:
    t_850, lat, lon, vn1 = load_imdaa_field("TMP", 850, TIMESTAMP, HOUR)
    t_500, _, _, _ = load_imdaa_field("TMP", 500, TIMESTAMP, HOUR)

    # Lapse rate proxy (K): larger positive = more unstable atmosphere
    lapse_rate = t_850 - t_500

    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    add_india_map_features(ax1)

    LON, LAT = np.meshgrid(lon, lat)
    p1 = ax1.pcolormesh(
        LON, LAT, lapse_rate,
        transform=ccrs.PlateCarree(),
        cmap='YlOrRd', shading='auto',
        vmin=np.nanpercentile(lapse_rate, 5),
        vmax=np.nanpercentile(lapse_rate, 95)
    )
    plt.colorbar(p1, ax=ax1, label='T_850 - T_500 (K)', fraction=0.046, pad=0.04, shrink=0.85)

    ax1.set_title(
        'Temperature Lapse Rate (CAPE Proxy)\n'
        'Higher = More Unstable Atmosphere',
        fontsize=11
    )

    print(f"   Variable: {vn1}")
    print(f"   Grid: {lapse_rate.shape} (lat={len(lat)}, lon={len(lon)})")
    print(f"   T_850 range: {np.nanmin(t_850):.1f} -- {np.nanmax(t_850):.1f} K")
    print(f"   T_500 range: {np.nanmin(t_500):.1f} -- {np.nanmax(t_500):.1f} K")
    print(f"   Lapse rate range: {np.nanmin(lapse_rate):.1f} -- {np.nanmax(lapse_rate):.1f} K")

    unstable_pct = 100 * np.sum(lapse_rate > 30) / lapse_rate.size
    print(f"   Pixels with lapse > 30K: {unstable_pct:.1f}%  (unstable)")

except FileNotFoundError as e:
    print(f"   WARNING: {e}")
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("TMP data not found")


# ----------------------------------------------------------
# Panel 2: Deep-Layer Bulk Wind Shear (300-850 hPa)
# ----------------------------------------------------------
print("\n[2/3] Computing Deep-Layer Wind Shear (300-850 hPa)...")

try:
    u_300, lat, lon, _ = load_imdaa_field("UGRD", 300, TIMESTAMP, HOUR)
    u_850, _, _, _ = load_imdaa_field("UGRD", 850, TIMESTAMP, HOUR)
    v_300, _, _, _ = load_imdaa_field("VGRD", 300, TIMESTAMP, HOUR)
    v_850, _, _, _ = load_imdaa_field("VGRD", 850, TIMESTAMP, HOUR)

    # Bulk shear vector magnitude (m/s)
    du = u_300 - u_850
    dv = v_300 - v_850
    shear_mag = np.sqrt(du**2 + dv**2)

    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    add_india_map_features(ax2)

    LON, LAT = np.meshgrid(lon, lat)
    p2 = ax2.pcolormesh(
        LON, LAT, shear_mag,
        transform=ccrs.PlateCarree(),
        cmap='hot_r', shading='auto',
        vmin=0, vmax=np.nanpercentile(shear_mag, 97)
    )
    plt.colorbar(p2, ax=ax2, label='Shear Magnitude (m/s)', fraction=0.046, pad=0.04, shrink=0.85)

    # Overlay shear vectors (every 20th gridpoint)
    skip = 20
    ax2.quiver(
        LON[::skip, ::skip], LAT[::skip, ::skip],
        du[::skip, ::skip], dv[::skip, ::skip],
        transform=ccrs.PlateCarree(),
        color='navy', alpha=0.5, scale=500, width=0.002
    )

    ax2.set_title(
        'Deep-Layer Bulk Wind Shear (300-850 hPa)\n'
        'Strong Shear -> Organized/Supercell Storms',
        fontsize=11
    )

    print(f"   Grid: {shear_mag.shape}")
    print(f"   Shear range: {np.nanmin(shear_mag):.1f} -- {np.nanmax(shear_mag):.1f} m/s")
    print(f"   Mean shear: {np.nanmean(shear_mag):.1f} m/s")
    severe_pct = 100 * np.sum(shear_mag > 20) / shear_mag.size
    print(f"   Pixels with shear > 20 m/s: {severe_pct:.1f}% (severe storm potential)")

except FileNotFoundError as e:
    print(f"   WARNING: {e}")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("UGRD/VGRD data not found")


# ----------------------------------------------------------
# Panel 3: Low-Level Convergence (925 hPa)
# ----------------------------------------------------------
print("\n[3/3] Computing Low-Level Convergence (925 hPa)...")

try:
    u_925, lat, lon, _ = load_imdaa_field("UGRD", 925, TIMESTAMP, HOUR)
    v_925, _, _, _ = load_imdaa_field("VGRD", 925, TIMESTAMP, HOUR)

    # Grid spacing: degrees -> approximate meters
    dlat = np.abs(lat[1] - lat[0])
    dlon = np.abs(lon[1] - lon[0])

    ref_lat_rad = np.deg2rad(25.0)
    dx = dlon * 111320.0 * np.cos(ref_lat_rad)
    dy = dlat * 110540.0

    # Convergence = -(dU/dx + dV/dy)
    # Positive = air converging = forced upward = storm trigger
    du_dx = np.gradient(u_925, dx, axis=1)
    dv_dy = np.gradient(v_925, dy, axis=0)
    convergence = -(du_dx + dv_dy)

    # Scale to 10^-5 s^-1 for readability
    conv_scaled = convergence * 1e5

    ax3 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
    add_india_map_features(ax3)

    LON, LAT = np.meshgrid(lon, lat)
    vmax_conv = min(float(np.nanpercentile(np.abs(conv_scaled), 97)), 50.0)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax_conv, vcenter=0, vmax=vmax_conv)

    p3 = ax3.pcolormesh(
        LON, LAT, conv_scaled,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r', norm=norm, shading='auto'
    )
    plt.colorbar(p3, ax=ax3, label='Convergence (x1e-5 s^-1)', fraction=0.046, pad=0.04, shrink=0.85)

    ax3.set_title(
        'Low-Level Convergence (925 hPa)\n'
        'Red = Convergence (Lift Trigger for Storm Initiation)',
        fontsize=11
    )

    print(f"   Grid: {convergence.shape}")
    print(f"   Grid spacing: dx={dx:.0f}m, dy={dy:.0f}m")
    print(f"   Convergence range: {np.nanmin(conv_scaled):.2f} -- {np.nanmax(conv_scaled):.2f} x1e-5 s^-1")

    conv_pct = 100 * np.sum(conv_scaled > 5) / conv_scaled.size
    print(f"   Strong convergence (>5x1e-5): {conv_pct:.1f}% of domain")

except FileNotFoundError as e:
    print(f"   WARNING: {e}")
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("UGRD/VGRD 925hPa data not found")


# ============================================================
# Save Output
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "imdaa_thermodynamics.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n" + "=" * 65)
print(f"  DONE -- Dashboard saved: {output_path}")
print("=" * 65)
print("\nKey Takeaways for Model Pipeline:")
print("  - Lapse rate (T850-T500) is a gridded CAPE proxy -- feeds instability head")
print("  - Deep-layer shear predicts storm organization & longevity")
print("  - 925 hPa convergence reveals the surface trigger for convection")
print("  -> These 3 fields (+ RH profiles) form the IMDAA feature set.\n")
