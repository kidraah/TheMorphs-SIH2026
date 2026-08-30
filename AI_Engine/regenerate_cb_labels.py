"""
SIH26077 — Regenerate Cloudburst Labels from HEM (Hydro-Estimator)
====================================================================
Replaces CB labels that were generated from GPI (L2G, 81x91 pixels, ~25km/pixel)
with labels from HEM (L2B, 2816x2805 pixels, ~1km/pixel).

Why HEM is better:
  GPI: 81x91 grid across all India = ~25km per pixel. A real cloudburst
       over 5km is averaged into a 25km cell and falls below the threshold.
       GPI max = 9mm — it is too coarse and clipped to see extreme events.

  HEM: 2816x2805 pixels at full disk resolution = ~1km per pixel.
       HEM max = 257mm — it captures the extreme localized events GPI misses.

Threshold: HEM > 25mm per 30-min = 50mm/hr = IMD cloudburst definition.

Run BEFORE regenerate_ff_labels.py (FF labels depend on CB labels).
"""

import os
import re
import glob
import json
import numpy as np
import h5py
from datetime import datetime, timedelta
from scipy.ndimage import binary_dilation

# ── Paths ─────────────────────────────────────────────────────────────────────
HEM_DIR     = 'C:/THEMORPHS/CONSOLIDATED_DATA/INSAT/L2B_HEM'
INDEX_PATH  = 'C:/THEMORPHS/dataset_root/window_index.json'

# Threshold: 25mm in 30min = 50mm/hr = IMD cloudburst definition
CB_THRESHOLD_MM = 25.0

# ── Build HEM file index by timestamp ─────────────────────────────────────────
def parse_insat_timestamp(fname):
    """Extract datetime from filename: 3RIMG_15AUG2019_0015_L2B_HEM_V01R00.h5"""
    m = re.search(r'3RIMG_(\d{2}[A-Z]+\d{4})_(\d{4})_', fname)
    if m:
        try:
            return datetime.strptime(m.group(1) + '_' + m.group(2), '%d%b%Y_%H%M')
        except ValueError:
            return None
    return None

print("Building HEM file index...")
hem_index = {}
for f in sorted(glob.glob(HEM_DIR + '/*.h5')):
    ts = parse_insat_timestamp(os.path.basename(f))
    if ts:
        hem_index[ts] = f

print(f"HEM files indexed: {len(hem_index)}")
if hem_index:
    ts_list = sorted(hem_index.keys())
    print(f"  Range: {ts_list[0]} to {ts_list[-1]}")


def get_nearest_hem(target_ts, tolerance_min=45):
    """Find nearest HEM file within ±tolerance minutes."""
    if not hem_index:
        return None
    nearest = min(hem_index.keys(), key=lambda t: abs((t - target_ts).total_seconds()))
    delta = abs((nearest - target_ts).total_seconds()) / 60
    if delta <= tolerance_min:
        return hem_index[nearest]
    return None


def generate_cb_from_hem(hem_path):
    """
    Load HEM file and create binary CB mask.

    HEM 'HEM' key shape: (1, 2816, 2805) — mm/30min rainfall rate.
    Fill values (negative or >500) are zeroed before thresholding.

    Returns:
        cb_mask: (1, 2816, 2805) float32 binary array
    """
    with h5py.File(hem_path, 'r') as f:
        hem_data = f['HEM'][:].squeeze().astype(np.float32)

    # Clean fill values
    hem_data[hem_data < 0]   = 0.0
    hem_data[hem_data > 500] = 0.0

    # Threshold to binary
    cb_mask = (hem_data >= CB_THRESHOLD_MM).astype(np.float32)

    # Morphological dilation: 3×3 kernel, 1 iteration
    # Expands sparse positive pixels into spatially coherent patches.
    # Turns ~25 pixels → ~200+ pixels per window.
    if cb_mask.sum() > 0:
        structure = np.ones((3, 3), dtype=bool)
        cb_mask = binary_dilation(cb_mask, structure=structure, iterations=1).astype(np.float32)

    return cb_mask[np.newaxis, :, :]   # (1, 2816, 2805)


# ── Main: iterate windows and regenerate CB labels ────────────────────────────
print("\nLoading window index...")
with open(INDEX_PATH) as f:
    windows = json.load(f)

print(f"Total windows: {len(windows)}")

updated  = 0
skipped  = 0
no_match = 0

for wi, win in enumerate(windows):
    imdaa_paths = win.get('imdaa_paths', [])
    if not imdaa_paths:
        skipped += 1
        continue

    # Extract base timestamp from first IMDAA file (e.g. 2019081600)
    m = re.search(r'_(\d{10})_', os.path.basename(imdaa_paths[0]))
    if not m:
        skipped += 1
        continue
    base_ts = datetime.strptime(m.group(1), '%Y%m%d%H')

    # Process every lead time
    for lead_str, target_paths in win.get('targets_by_lead', {}).items():
        cb_path = target_paths.get('cloudburst', '').replace('\\', '/')
        if not cb_path or not os.path.exists(cb_path):
            skipped += 1
            continue

        target_ts = base_ts + timedelta(hours=int(lead_str))
        hem_file  = get_nearest_hem(target_ts)

        if hem_file is None:
            no_match += 1
            # Keep the existing label rather than zero-filling
            continue

        try:
            cb_mask = generate_cb_from_hem(hem_file)
            np.save(cb_path, cb_mask)
            updated += 1
        except Exception as e:
            print(f"  ERROR on window {wi}, lead {lead_str}: {e}")
            skipped += 1

    if (wi + 1) % 20 == 0:
        print(f"  Progress: {wi+1}/{len(windows)} windows done...")

# ── Final report ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CB LABEL REGENERATION COMPLETE")
print(f"  Updated  : {updated}")
print(f"  Skipped  : {skipped} (path missing or parse error)")
print(f"  No HEM   : {no_match} (no HEM file within ±45 min)")
print("="*60)

# Quick pixel density check on a sample label
import torch
from data_loader import resize_binary_target

sample_paths = []
for win in windows[:5]:
    for lead, tgt in win.get('targets_by_lead', {}).items():
        p = tgt.get('cloudburst', '').replace('\\', '/')
        if p and os.path.exists(p):
            sample_paths.append(p)
            break

print("\nSample CB label density check (after regen):")
total_px = 0
for p in sample_paths[:5]:
    arr = np.load(p)
    pos = (arr > 0.5).sum()
    total_px += pos
    print(f"  {os.path.basename(os.path.dirname(p))}: {pos} positive pixels / {arr.size} total")

if sample_paths:
    avg = total_px / len(sample_paths[:5])
    pct = 100.0 * avg / (256 * 256)
    print(f"  Avg positive (at native res): {avg:.1f} px")
    print(f"  After resize to 256x256 (nearest-neighbor): ~{pct:.3f}% positive")
    if avg < 5:
        print("  WARNING: Still very sparse. Consider raising dilation to iterations=2.")
    elif avg > 500:
        print("  WARNING: Unusually dense. Check HEM threshold (currently 25mm/30min).")
    else:
        print("  OK: Density looks learnable.")
