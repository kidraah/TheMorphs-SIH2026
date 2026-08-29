"""
Script 1: INSAT-3DR Satellite Signature Analysis
SIH26077 -- Exploratory Data Analysis & Precursor Extraction

Analyzes four key satellite-derived atmospheric signatures:
  1. Cloud Top Temperature (CTT) Drop Rate -- explosive updraft detection
  2. Water Vapor (WV) Brightness Temperature -- moisture pool identification
  3. GOES Precipitation Index (GPI) -- satellite-derived QPE time-series
  4. Humidity/Emissivity Map (HEM) -- atmospheric moisture proxy

Data:  INSAT-3D/3DR products from dataset_root/insat/
Output: PNG saved to output/insat_signatures.png
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (must be before pyplot import)

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
ROOT = "C:/THEMORPHS/dataset_root"
OUTPUT_DIR = "C:/THEMORPHS/output"
FOCUS_DATE = "16AUG"
FOCUS_YEAR = "2019"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Data Loading Functions
# ============================================================

def decode_latlon(arr):
    """Decode INSAT int16/int32 lat/lon arrays to float degrees.
    INSAT products store lat/lon as integers scaled by 100."""
    out = arr.astype(np.float64)
    if np.nanmax(np.abs(out)) > 360:
        out /= 100.0
    return out


def load_ctp_pair(filepath):
    """Load CTT, CTP, and coordinates from an L2B CTP product file."""
    with h5py.File(filepath, 'r') as f:
        ctt = np.squeeze(f['CTT'][:])
        ctp = np.squeeze(f['CTP'][:])
        lat = decode_latlon(f['Latitude'][:])
        lon = decode_latlon(f['Longitude'][:])

        # INSAT fill value is -999
        ctt = np.ma.masked_where((ctt < 0) | (ctt > 400), ctt)
        ctp = np.ma.masked_where((ctp < 0) | (ctp > 1100), ctp)

    return ctt, ctp, lat, lon


def load_wv_brightness_temp(filepath, stride=4):
    """Convert Water Vapor grey counts to brightness temperature via LUT."""
    with h5py.File(filepath, 'r') as f:
        wv_raw = np.squeeze(f['IMG_WV'][:])
        wv_lut = f['IMG_WV_TEMP'][:]
        lat_wv = decode_latlon(f['Latitude_WV'][:])
        lon_wv = decode_latlon(f['Longitude_WV'][:])

        wv_clipped = np.clip(wv_raw, 0, len(wv_lut) - 1)
        wv_temp = wv_lut[wv_clipped]
        wv_temp = np.ma.masked_where((wv_temp <= 0) | (wv_temp > 400), wv_temp)

        # Subsample for performance
        wv_temp = wv_temp[::stride, ::stride]
        lat_wv = lat_wv[::stride, ::stride]
        lon_wv = lon_wv[::stride, ::stride]

    return wv_temp, lat_wv, lon_wv


def load_gpi_timeseries(gpi_dir):
    """Build GPI precipitation time-series across all available files."""
    files = sorted(glob.glob(os.path.join(gpi_dir, "*.h5")))
    timestamps = []
    mean_precip = []
    max_precip = []

    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.split('_')
        try:
            dt = datetime.strptime(f"{parts[1]}_{parts[2]}", "%d%b%Y_%H%M")
            timestamps.append(dt)
        except (ValueError, IndexError):
            continue

        with h5py.File(fpath, 'r') as f:
            gpi = np.squeeze(f['GPI'][:])
            gpi_valid = gpi[gpi > 0]
            mean_precip.append(float(np.mean(gpi_valid)) if len(gpi_valid) > 0 else 0.0)
            max_precip.append(float(np.max(gpi_valid)) if len(gpi_valid) > 0 else 0.0)

    return timestamps, mean_precip, max_precip


def load_hem_snapshot(filepath, stride=8):
    """Load Humidity/Emissivity Map, subsampled for plotting."""
    with h5py.File(filepath, 'r') as f:
        hem = np.squeeze(f['HEM'][:])
        lat = decode_latlon(f['Latitude'][:])
        lon = decode_latlon(f['Longitude'][:])

        hem = np.ma.masked_where((hem < 0) | (hem > 500), hem)

        hem = hem[::stride, ::stride]
        lat = lat[::stride, ::stride]
        lon = lon[::stride, ::stride]

    return hem, lat, lon


# ============================================================
# Main Analysis
# ============================================================

print("=" * 65)
print("  SCRIPT 1: INSAT-3DR SATELLITE SIGNATURE ANALYSIS")
print("  SIH26077 -- Severe Weather Precursor Extraction")
print("=" * 65)

fig = plt.figure(figsize=(22, 18))
gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
fig.suptitle(
    'INSAT-3DR Satellite Signatures -- August 2019\n'
    'SIH26077: Hyper-Local Early Warning for Severe Weather',
    fontsize=17, fontweight='bold', y=0.98
)


# ----------------------------------------------------------
# Panel 1: CTT Drop Rate (dCTT/dt between consecutive passes)
# ----------------------------------------------------------
print("\n[1/4] Computing CTT Drop Rate...")

ctp_dir = f"{ROOT}/insat/3RIMG_L2B_CTP/{FOCUS_YEAR}/{FOCUS_DATE}"
ctp_files = sorted(glob.glob(os.path.join(ctp_dir, "*.h5")))

ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#333')
ax1.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', color='#666')

if len(ctp_files) >= 2:
    ctt_t1, _, lat_ctp, lon_ctp = load_ctp_pair(ctp_files[0])
    ctt_t2, _, _, _ = load_ctp_pair(ctp_files[1])

    # dCTT/dt in K/hr (negative = cooling = explosive updraft)
    delta_t_hours = 3.0
    ctt_drop_rate = (ctt_t2 - ctt_t1) / delta_t_hours

    vmax = min(float(np.nanmax(np.abs(ctt_drop_rate))), 20.0)
    vmax = max(vmax, 1.0)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    p1 = ax1.pcolormesh(
        lon_ctp, lat_ctp, ctt_drop_rate,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm_r', norm=norm, shading='auto'
    )
    plt.colorbar(p1, ax=ax1, label='dCTT/dt (K/hr)', fraction=0.046, pad=0.04)

    t1_label = os.path.basename(ctp_files[0]).split('_')[2]
    t2_label = os.path.basename(ctp_files[1]).split('_')[2]
    ax1.set_title(
        f"CTT Drop Rate -- {FOCUS_DATE} ({t1_label} -> {t2_label} UTC)\n"
        f"Blue = Rapid Cooling (Vertical Updraft Signature)",
        fontsize=11
    )

    cooling_mask = ctt_drop_rate < -5
    print(f"   Time pair: {t1_label} -> {t2_label} UTC")
    print(f"   Grid size: {ctt_drop_rate.shape}")
    print(f"   Max cooling rate: {float(np.nanmin(ctt_drop_rate)):.2f} K/hr")
    print(f"   Pixels with cooling > 5 K/hr: {int(np.sum(cooling_mask))}")
else:
    ax1.set_title("Insufficient CTP files for drop rate calculation")
    print("   WARNING: Need at least 2 consecutive CTP files")


# ----------------------------------------------------------
# Panel 2: Water Vapor Brightness Temperature
# ----------------------------------------------------------
print("\n[2/4] Loading Water Vapor channel...")

l1b_dir = f"{ROOT}/insat/3RIMG_L1B_STD/{FOCUS_YEAR}/{FOCUS_DATE}"
l1b_files = sorted(glob.glob(os.path.join(l1b_dir, "*.h5")))

ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#333')
ax2.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', color='#666')

if l1b_files:
    wv_temp, lat_wv, lon_wv = load_wv_brightness_temp(l1b_files[0], stride=4)

    p2 = ax2.pcolormesh(
        lon_wv, lat_wv, wv_temp,
        transform=ccrs.PlateCarree(),
        cmap='BuPu_r', shading='auto'
    )
    plt.colorbar(p2, ax=ax2, label='WV Brightness Temperature (K)', fraction=0.046, pad=0.04)

    wv_label = os.path.basename(l1b_files[0]).split('_')[2]
    ax2.set_title(
        f"Water Vapor Channel -- {FOCUS_DATE} {wv_label} UTC\n"
        f"Low BT = High Moisture / Deep Convection (IWV Proxy)",
        fontsize=11
    )

    print(f"   File: {os.path.basename(l1b_files[0])}")
    print(f"   WV BT range: {float(np.nanmin(wv_temp)):.1f} -- {float(np.nanmax(wv_temp)):.1f} K")
    print(f"   Grid size (subsampled): {wv_temp.shape}")
else:
    ax2.set_title("No L1B STD files found")
    print("   WARNING: No L1B files available")


# ----------------------------------------------------------
# Panel 3: GPI Precipitation Time-Series
# ----------------------------------------------------------
print("\n[3/4] Building GPI precipitation time series...")

gpi_dir = f"{ROOT}/insat/3RIMG_L2G_GPI"
timestamps, mean_precip, max_precip = load_gpi_timeseries(gpi_dir)

ax3 = fig.add_subplot(gs[1, 0])

if timestamps:
    x = range(len(timestamps))

    ax3.fill_between(x, max_precip, alpha=0.15, color='#e74c3c', label='Max GPI')
    ax3.plot(x, max_precip, color='#e74c3c', linewidth=1.0, alpha=0.6)
    ax3.fill_between(x, mean_precip, alpha=0.3, color='#2980b9')
    ax3.plot(x, mean_precip, color='#2980b9', linewidth=1.5, marker='o', markersize=3, label='Mean GPI')

    tick_positions = list(range(0, len(timestamps), 6))
    tick_labels = [timestamps[i].strftime('%d/%m\n%H:%M') for i in tick_positions]
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels, fontsize=7)

    ax3.set_xlabel('Time', fontsize=10)
    ax3.set_ylabel('GPI Precipitation (mm/hr)', fontsize=10)
    ax3.set_title(
        'Satellite-Derived QPE (GOES Precipitation Index)\n'
        'Aug 15-25, 2019 -- All Available 3-Hourly Passes',
        fontsize=11
    )
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(fontsize=9, loc='upper right')

    print(f"   Total GPI files: {len(timestamps)}")
    print(f"   Date range: {timestamps[0].strftime('%Y-%m-%d %H:%M')} -> {timestamps[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"   Peak mean GPI: {max(mean_precip):.2f} mm/hr")
    print(f"   Peak max GPI:  {max(max_precip):.2f} mm/hr")
else:
    ax3.set_title("No GPI data found")
    print("   WARNING: No GPI files available")


# ----------------------------------------------------------
# Panel 4: Humidity/Emissivity Map (HEM)
# ----------------------------------------------------------
print("\n[4/4] Loading Humidity/Emissivity Map...")

hem_dir = f"{ROOT}/insat/3RIMG_L2B_HEM/{FOCUS_YEAR}/{FOCUS_DATE}"
hem_files = sorted(glob.glob(os.path.join(hem_dir, "*.h5")))

ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
ax4.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#333')
ax4.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', color='#666')

if hem_files:
    hem, lat_hem, lon_hem = load_hem_snapshot(hem_files[0], stride=8)

    p4 = ax4.pcolormesh(
        lon_hem, lat_hem, hem,
        transform=ccrs.PlateCarree(),
        cmap='YlGnBu', shading='auto'
    )
    plt.colorbar(p4, ax=ax4, label='HEM Value', fraction=0.046, pad=0.04)

    ax4.set_title(
        f"Humidity/Emissivity Map -- {FOCUS_DATE}\n"
        f"Atmospheric Moisture Distribution (IWV Complement)",
        fontsize=11
    )

    print(f"   File: {os.path.basename(hem_files[0])}")
    print(f"   HEM range: {float(np.nanmin(hem)):.3f} -- {float(np.nanmax(hem)):.3f}")
    print(f"   Grid size (subsampled): {hem.shape}")
else:
    ax4.set_title("No HEM files found")
    print("   WARNING: No HEM files available")


# ============================================================
# Save Output
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "insat_signatures.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n" + "=" * 65)
print(f"  DONE -- Dashboard saved: {output_path}")
print("=" * 65)
print("\nKey Takeaways for Model Pipeline:")
print("  - CTT drop rate reveals explosive updraft signatures")
print("  - WV channel tracks moisture pool concentration (IWV proxy)")
print("  - GPI provides satellite-derived rainfall validation")
print("  - HEM captures atmospheric moisture/emissivity distribution")
print("  -> All 4 are INSAT-derived features for the shared backbone.\n")
