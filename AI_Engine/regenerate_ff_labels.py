"""
regenerate_ff_labels.py
=======================
Regenerates Flash Flood labels derived from Cloudburst labels + DEM slope.

Physical definition (per ps.md):
  Flash Flood = Cloudburst (QPE > 50mm/3hr) occurring over steep terrain
                that channels runoff into river valleys.

  FF label = 1  where  CB label = 1  AND  terrain slope > SLOPE_THRESH_DEG

Why this is correct:
  - Cloudburst labels already encode "extreme precipitation happened here"
  - DEM slope encodes where that precipitation will cause destructive runoff
  - Their intersection = flash flood risk zone
  - This is exactly what ps.md says: "overlay atmospheric maps onto DEM to
    calculate how terrain slope channels extreme precipitation"

Run once before training:
  python regenerate_ff_labels.py
"""

import os, sys, glob
import numpy as np
import rasterio
from scipy.ndimage import zoom

# Script lives in scripts/ — add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_ROOT, DEM_PATH, GRID_SIZE

TARGETS_DIR    = os.path.join(DATASET_ROOT, 'targets')
SLOPE_THRESH   = 10.0   # degrees — pixels steeper than this can cause flash floods
                         # Uttarakhand avg slope is 15-30°, so 10° is conservative


def compute_slope(dem_path, grid_size):
    """Load CartoDEM, compute slope in degrees, resize to model grid."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        dem[dem < -100] = 0.0                     # fill nodata
        pixel_deg = abs(src.transform.a)           # degrees per pixel
        pixel_m   = pixel_deg * 111_000            # ~111km per degree latitude

    # Finite-difference gradient (dz/dx and dz/dy in m/m)
    dy, dx = np.gradient(dem, pixel_m, pixel_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Resize to model grid (256×256)
    zy = grid_size[0] / slope_deg.shape[0]
    zx = grid_size[1] / slope_deg.shape[1]
    slope_resized = zoom(slope_deg, (zy, zx), order=1)
    return slope_resized.astype(np.float32)


def load_cb_binary(cb_path, grid_size):
    """Load cloudburst.npy, apply QPE threshold if needed, resize."""
    cb = np.load(cb_path).squeeze().astype(np.float32)
    if cb.max() > 10.0:                           # raw QPE in mm → binarize
        cb = (cb >= 50.0).astype(np.float32)
    if cb.shape != tuple(grid_size):
        zy = grid_size[0] / cb.shape[0]
        zx = grid_size[1] / cb.shape[1]
        cb = zoom(cb, (zy, zx), order=1)
    return (cb > 0.5).astype(np.float32)


def main():
    thresh = SLOPE_THRESH   # local copy — may be lowered if domain is flat

    print("\n" + "=" * 55)
    print("  Flash Flood Label Regeneration")
    print(f"  Slope threshold : {thresh}°")
    print(f"  Targets dir     : {TARGETS_DIR}")
    print("=" * 55)

    # ── Step 1: DEM slope mask ───────────────────────────────
    print("\n[1/3] Computing DEM slope map...")
    slope = compute_slope(DEM_PATH, GRID_SIZE)
    slope_mask = (slope > thresh).astype(np.float32)
    steep_pct = slope_mask.mean() * 100
    slope_max  = slope.max()
    print(f"      Max slope in domain : {slope_max:.1f}°")
    print(f"      Pixels > {thresh}°        : {steep_pct:.1f}% of grid")

    if steep_pct < 1.0:
        print("\n  [!] Less than 1% steep pixels — lowering threshold to 5°.")
        thresh     = 5.0
        slope_mask = (slope > thresh).astype(np.float32)
        steep_pct  = slope_mask.mean() * 100
        print(f"      Pixels > {thresh}°        : {steep_pct:.1f}%")

    # ── Step 2: Process every window/lead folder ─────────────
    folders = sorted(glob.glob(os.path.join(TARGETS_DIR, 'win_*_lead_*')))
    print(f"\n[2/3] Processing {len(folders)} window/lead folders...")

    total_ff = 0
    total_cb = 0
    updated  = 0

    for folder in folders:
        cb_path = os.path.join(folder, 'cloudburst.npy')
        ff_path = os.path.join(folder, 'flash_flood.npy')

        if not os.path.exists(cb_path):
            continue

        cb = load_cb_binary(cb_path, GRID_SIZE)
        ff = ((cb > 0.5) & (slope_mask > 0.5)).astype(np.float32)

        total_cb += int(cb.sum())
        total_ff += int(ff.sum())

        np.save(ff_path, ff)
        updated += 1

    # ── Step 3: Report ────────────────────────────────────────
    print(f"\n[3/3] Done — {updated} flash_flood.npy files written.")
    print(f"      CB positive pixels (total) : {total_cb:,}")
    print(f"      FF positive pixels (total) : {total_ff:,}")

    if total_ff == 0:
        print("\n  [!] Still 0 FF pixels. CB labels may themselves be empty.")
        print("      Check that cloudburst.npy files have max > 50 mm.")
    else:
        ratio = total_ff / max(total_cb, 1)
        print(f"      FF/CB ratio                : {ratio:.2f}  (expect 0.3–0.8)")
        print("\n  [OK] Flash Flood labels generated successfully.")
        print("       Now run: python train.py")


if __name__ == '__main__':
    main()

