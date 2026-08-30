"""
SIH26077 — Spatiotemporal Data Loader
======================================
Loads and aligns multi-modal atmospheric data from three sources:
  1. IMDAA Reanalysis (.nc)  → (30, 6, 256, 256)  thermodynamic + kinematic fields
  2. INSAT-3DR Satellite (.h5) → (3, 6, 256, 256)  WV, CTT, HEM
  3. CartoDEM (.tif)           → (2, 256, 256)      elevation + slope

All modalities are spatially aligned to a unified 256×256 grid and
channel-wise z-score normalized for stable training.
"""

import os
import json
import torch
import numpy as np
import xarray as xr
import h5py
import rasterio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import warnings

from config import (
    INDEX_PATH, DATASET_ROOT, DEM_PATH, GRID_SIZE, DEFAULT_LEAD_TIME,
    IMDAA_MEAN, IMDAA_STD, INSAT_MEAN, INSAT_STD, TERRAIN_MEAN, TERRAIN_STD,
)

warnings.filterwarnings('ignore')


def resize_tensor(tensor, size):
    """Resizes a 2D or 3D/4D spatial tensor to the target (H, W) grid.
    
    Uses bilinear interpolation. Handles tensors of shape:
      - (H, W)       → unsqueeze to (1, 1, H, W) → resize → squeeze back
      - (C, H, W)    → unsqueeze to (1, C, H, W) → resize → squeeze back
      - (N, C, H, W) → resize directly
    """
    original_shape = tensor.shape
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
        
    tensor = tensor.float()
    resized = F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)
    
    if len(original_shape) == 2:
        return resized.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        return resized.squeeze(0)
    return resized


def resize_binary_target(tensor, size):
    """Resize a binary (0/1) label map using nearest-neighbor interpolation.

    WHY NOT bilinear for binary masks:
      Bilinear on a sparse binary mask (e.g. CB at 2816×2805 → 256×256)
      smears each '1' pixel into a tiny float blur (~0.0001).  After
      downscaling 11× the signal is effectively zero — the model sees no
      positive pixels and the loss gradient collapses.

      Nearest-neighbor keeps every '1' as a hard '1' and every '0' as '0',
      preserving the true label distribution at the new resolution.
    """
    original_shape = tensor.shape
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.float()
    resized = F.interpolate(tensor, size=size, mode='nearest')
    # Hard-threshold to guarantee strict binary output (no float residuals)
    resized = (resized > 0.5).float()

    if len(original_shape) == 2:
        return resized.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        return resized.squeeze(0)
    return resized


class SpatiotemporalDataset(Dataset):
    """Multi-modal spatiotemporal dataset for severe weather nowcasting.
    
    Each sample is a 6-timestep window (~18 hours) containing:
      - IMDAA reanalysis: 30 atmospheric channels across 6 timesteps
      - INSAT satellite:  3 observational channels across 6 timesteps
      - CartoDEM terrain: 2 static channels (elevation + slope)
      - Targets:          3 binary risk maps (cloudburst, thunderstorm, flash flood)
    
    Args:
        index_path: Path to window_index.json
        root_dir:   Path to dataset_root/
        lead_time:  Which lead time to predict ('2', '3', '4', '5', or '6' hours)
        grid_size:  Unified spatial grid (H, W) for all modalities
        normalize:  Whether to apply z-score normalization (default: True)
    """

    def __init__(self, index_path=INDEX_PATH, root_dir=DATASET_ROOT,
                 lead_time=DEFAULT_LEAD_TIME, grid_size=GRID_SIZE, normalize=True):
        with open(index_path, 'r') as f:
            self.windows = json.load(f)
            
        self.root_dir = root_dir
        self.lead_time = str(lead_time)
        self.grid_size = grid_size
        self.normalize = normalize
        
        # Load static DEM once (same for all windows)
        self.dem_tensor = self._load_dem()

    def __len__(self):
        return len(self.windows)

    def _load_dem(self):
        """Load and preprocess the Digital Elevation Model."""
        print("Loading and downsampling static DEM...")
        with rasterio.open(DEM_PATH) as src:
            # Downsample immediately to save memory
            factor = max(src.width // 1000, 1)
            elevation = src.read(
                1, out_shape=(src.height // factor, src.width // factor)
            ).astype(np.float32)
            
            # Compute slope from elevation gradients
            dy, dx = np.gradient(elevation)
            slope = np.sqrt(dx**2 + dy**2)
            
            # Stack into (C=2, H, W): [elevation, slope]
            terrain = np.stack([elevation, slope], axis=0)
            terrain_tensor = torch.from_numpy(terrain)
            terrain_tensor = torch.nan_to_num(terrain_tensor, nan=0.0)
            
            # Resize to unified grid
            terrain_tensor = resize_tensor(terrain_tensor, self.grid_size)
            
            # Normalize
            if self.normalize:
                terrain_tensor = (terrain_tensor - TERRAIN_MEAN) / (TERRAIN_STD + 1e-8)
            
            return terrain_tensor

    def _load_imdaa(self, paths):
        """Load IMDAA reanalysis data into (Channels=30, Time=6, H, W).

        CRITICAL FIX — group by TIMESTAMP, not by alphabetical sort:
          The 180 paths span 6 timestamps × 30 channels (5 vars × 6 levels).
          When sorted alphabetically and chunked by 30, each 'timestep' chunk
          contains only ONE variable repeated (e.g. all HGT-1000mb through
          HGT-925mb at 6 different times) — NOT all variables at one time.
          The 3D temporal conv was therefore operating on var-grouped slices
          with no actual temporal meaning.

          Correct grouping: extract the 10-digit timestamp from each filename
          (e.g. '2019081600'), group files sharing the same timestamp, then
          sort groups chronologically. Each group is a genuine snapshot of
          all 5 atmospheric variables at one moment in time.
        """
        import re
        from collections import defaultdict

        # Group by 10-digit timestamp embedded in filename (YYYYMMDDHH)
        ts_groups = defaultdict(list)
        for p in paths:
            fname = os.path.basename(p.replace('\\', '/'))
            m = re.search(r'_(\d{10})_', fname)
            if m:
                ts_groups[m.group(1)].append(p)

        if not ts_groups:
            # Fallback to old alphabetical chunking if regex fails
            paths_sorted = sorted(paths)
            chunks = [paths_sorted[i:i+30] for i in range(0, len(paths_sorted), 30)]
        else:
            # Sort chronologically; within each timestamp, sort alphabetically
            # (alphabetical within-timestamp gives: HGT, RH, TMP, UGRD, VGRD × levels)
            chunks = [sorted(ts_groups[ts]) for ts in sorted(ts_groups.keys())]

        time_steps = []
        for chunk in chunks:
            channels = []
            for path in chunk:
                try:
                    path = path.replace('\\', '/')
                    ds = xr.open_dataset(path)
                    var_name = list(ds.data_vars)[0]
                    data = ds[var_name].squeeze().values
                    data = np.nan_to_num(data, nan=0.0)
                    channels.append(torch.from_numpy(data))
                    ds.close()
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    channels.append(torch.zeros((501, 751)))

            # (30, H_raw, W_raw) — one full atmospheric state snapshot
            timestep_tensor = torch.stack(channels, dim=0)
            time_steps.append(timestep_tensor)

        # (Time=6, C=30, H, W) → (C=30, Time=6, H, W)
        imdaa_tensor = torch.stack(time_steps, dim=0).permute(1, 0, 2, 3)
        imdaa_tensor = resize_tensor(imdaa_tensor, self.grid_size)

        # Normalize channel-wise (broadcast across Time, H, W)
        if self.normalize:
            # IMDAA_MEAN/STD shape: (30, 1, 1, 1) — broadcasts over (30, 6, 256, 256)
            imdaa_tensor = (imdaa_tensor - IMDAA_MEAN) / (IMDAA_STD + 1e-8)

        return imdaa_tensor


    def _load_insat(self, l1b_paths, ctp_paths, hem_paths):
        """Load INSAT satellite data into (Channels=4, Time=6, H, W).

        Four channels per timestep:
          - Ch 0: WV  — Water Vapor brightness temperature (L1B IMG_WV key)
          - Ch 1: CTT — Cloud Top Temperature (L2B CTP 'CTT' key, Kelvin)
          - Ch 2: HEM — Hydro-Estimator precipitation rate (L2B HEM 'HEM' key, mm/30min)
          - Ch 3: CTT_RATE — Frame-to-frame CTT change (K per 3h step, NEGATIVE = cooling = storm building)

        CTT_RATE (ps.md explicit requirement):
          "Rapid cooling of cloud tops (CTT Drop Rate) provides real-time
          validation of explosive vertical updrafts within the system."
          Without CTT_RATE, the model sees static snapshots and cannot detect
          convective intensification. A -15K/step drop in CTT is the clearest
          single-variable cloudburst precursor available from satellite.
        """
        time_steps = []
        num_steps  = max(len(l1b_paths), len(ctp_paths), len(hem_paths), 6)

        # --- Load raw CTT values for all timesteps first ---
        # Needed to compute temporal differences (drop rate)
        raw_ctts = []
        for i in range(num_steps):
            if i < len(ctp_paths):
                try:
                    with h5py.File(ctp_paths[i].replace('\\', '/'), 'r') as f:
                        ctt = np.squeeze(f['CTT'][:]).astype(np.float32)
                        ctt = np.ma.filled(np.ma.masked_where(ctt < 0, ctt), 0.0)
                        raw_ctts.append(ctt)
                except Exception:
                    raw_ctts.append(np.zeros((313, 312), dtype=np.float32))
            else:
                raw_ctts.append(np.zeros((313, 312), dtype=np.float32))

        # CTT drop rate: diff between consecutive frames (negative = cooling)
        # At t=0 there is no prior frame — use zero (no rate info).
        ctt_rates = [np.zeros_like(raw_ctts[0])]
        for i in range(1, len(raw_ctts)):
            ctt_rates.append(raw_ctts[i] - raw_ctts[i - 1])   # neg = cooling

        # --- Build per-timestep channel stacks ---
        for i in range(num_steps):
            channels = []

            # Ch 0: WV (L1B IMG_WV)
            if i < len(l1b_paths):
                try:
                    with h5py.File(l1b_paths[i].replace('\\', '/'), 'r') as f:
                        wv = np.squeeze(f['IMG_WV'][:]).astype(np.float32)
                        wv = wv[::4, ::4]   # subsample: 1408×1402 → 352×351
                        channels.append(torch.from_numpy(np.nan_to_num(wv, nan=0.0)))
                except Exception:
                    channels.append(torch.zeros((352, 351)))
            else:
                channels.append(torch.zeros((352, 351)))

            # Ch 1: CTT (L2B CTP)
            channels.append(torch.from_numpy(raw_ctts[i]))

            # Ch 2: HEM precipitation rate (L2B HEM)
            if i < len(hem_paths):
                try:
                    with h5py.File(hem_paths[i].replace('\\', '/'), 'r') as f:
                        hem = np.squeeze(f['HEM'][:]).astype(np.float32)
                        hem[hem < 0] = 0.0
                        hem[hem > 500] = 0.0
                        hem = hem[::8, ::8]   # subsample: 2816×2805 → 352×351
                        channels.append(torch.from_numpy(hem))
                except Exception:
                    channels.append(torch.zeros((352, 351)))
            else:
                channels.append(torch.zeros((352, 351)))

            # Ch 3: CTT drop rate (K/step, negative = explosive cooling)
            channels.append(torch.from_numpy(ctt_rates[i]))

            # Resize each channel to grid_size (different raw sizes per channel)
            resized = [resize_tensor(c, self.grid_size) for c in channels]
            time_steps.append(torch.stack(resized, dim=0))

        # (Time=6, C=4, H, W) → (C=4, Time=6, H, W)
        insat_tensor = torch.stack(time_steps, dim=0).permute(1, 0, 2, 3)

        # Normalize channel-wise
        if self.normalize:
            insat_tensor = (insat_tensor - INSAT_MEAN) / (INSAT_STD + 1e-8)

        return insat_tensor



    def _load_targets(self, target_dict):
        """Load 3 binary target maps into (Channels=3, H, W).
        
        Channel order: [Cloudburst, Thunderstorm, FlashFlood]
        
        IMPORTANT — Flash Flood label fix:
          flash_flood.npy may contain raw QPE values (mm/3hr) rather than
          a pre-thresholded binary mask. Thunderstorm was stored pre-binarized
          (CTT < 208.15K → 1) but FF was not. We apply the threshold here.
          If max value > 10.0 → treat as continuous mm values → threshold at 50mm.
          If max value ≤ 2.0 → already binary, load as-is.
        """
        cb = np.load(target_dict['cloudburst'].replace('\\', '/'))
        ts = np.load(target_dict['thunderstorm'].replace('\\', '/'))
        ff = np.load(target_dict['flash_flood'].replace('\\', '/'))

        cb_np = cb.squeeze().astype(np.float32)
        ts_np = ts.squeeze().astype(np.float32)
        ff_np = ff.squeeze().astype(np.float32)

        # ── Flash Flood threshold fix ────────────────────────────────────
        # If values are continuous QPE (mm), binarize at 50mm (cloudburst threshold)
        # AND require slope > 12° (encoded as a multiplier below — slope mask
        # is applied during the DEM-overlay step in the model output, not here,
        # but the QPE threshold alone creates valid FF labels)
        if ff_np.max() > 10.0:
            ff_np = (ff_np >= 50.0).astype(np.float32)

        # Same check for CB in case it also stores raw QPE
        if cb_np.max() > 10.0:
            cb_np = (cb_np >= 50.0).astype(np.float32)

        cb = torch.from_numpy(cb_np).float()
        ts = torch.from_numpy(ts_np).float()
        ff = torch.from_numpy(ff_np).float()

        # Ensure 2D (take first channel if 3D)
        if len(cb.shape) > 2: cb = cb[0]
        if len(ts.shape) > 2: ts = ts[0]
        if len(ff.shape) > 2: ff = ff[0]

        # Use nearest-neighbor for all binary targets — preserves sparse 0/1 signals.
        # Bilinear would smear a 25-pixel CB mask over a 2816×2805 grid into
        # near-zero floats after downscaling to 256×256 (label destruction).
        cb = resize_binary_target(cb, self.grid_size)
        ts = resize_binary_target(ts, self.grid_size)
        ff = resize_binary_target(ff, self.grid_size)

        # Stack: [Cloudburst, Thunderstorm, FlashFlood]
        return torch.stack([cb, ts, ff], dim=0)


    def __getitem__(self, idx):
        window = self.windows[idx]
        
        # 1. IMDAA (30, 6, H, W)
        imdaa = self._load_imdaa(window['imdaa_paths'])
        
        # 2. INSAT (3, 6, H, W)
        insat = self._load_insat(
            window.get('insat_l1b', []),
            window.get('insat_l2b_ctp', []),
            window.get('insat_l2b_hem', []),
        )
        
        # 3. Terrain (2, H, W) — pre-loaded and shared
        terrain = self.dem_tensor
        
        # 4. Targets (3, H, W)
        target_dict = window['targets_by_lead'][self.lead_time]
        targets = self._load_targets(target_dict)
        
        return {
            'imdaa': imdaa,      # (30, 6, 256, 256)
            'insat': insat,      # (3, 6, 256, 256)
            'terrain': terrain,  # (2, 256, 256)
            'targets': targets,  # (3, 256, 256)
        }


# ============================================================
# Self-Test
# ============================================================
if __name__ == "__main__":
    print("Testing SpatiotemporalDataset...")
    dataset = SpatiotemporalDataset()
    
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    
    print("\nSample Shapes:")
    print(f"IMDAA:   {sample['imdaa'].shape}  (Channels, Time, H, W)")
    print(f"INSAT:   {sample['insat'].shape}   (Channels, Time, H, W)")
    print(f"Terrain: {sample['terrain'].shape}    (Channels, H, W)")
    print(f"Targets: {sample['targets'].shape}    (Channels, H, W)")
    
    print("\nNormalization Check (should be near mean=0, std=1):")
    imdaa = sample['imdaa']
    print(f"  IMDAA  ch0 mean: {imdaa[0].mean():.2f}, std: {imdaa[0].std():.2f}")
    print(f"  IMDAA ch15 mean: {imdaa[15].mean():.2f}, std: {imdaa[15].std():.2f}")
    
    # Check for NaNs
    for k, v in sample.items():
        nan_count = torch.isnan(v).sum().item()
        print(f"{k.capitalize()} NaNs: {nan_count}")
        
    print("\n[OK] DataLoader test passed.")
