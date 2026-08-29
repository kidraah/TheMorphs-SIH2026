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
        
        180 .nc file paths are grouped into 6 chunks of 30 (one per timestep).
        Each chunk contains 5 variables × 6 pressure levels = 30 channels.
        """
        paths = sorted(paths)
        time_steps = []
        
        # Group into 30-file chunks (1 chunk = 1 timestep)
        chunks = [paths[i:i+30] for i in range(0, len(paths), 30)]
        
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
                    
            # (30, H_raw, W_raw)
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
        """Load INSAT satellite data into (Channels=3, Time=6, H, W).
        
        Three channels per timestep:
          - WV: Water Vapor brightness temperature (from L1B)
          - CTT: Cloud Top Temperature (from L2B CTP product)
          - HEM: Humidity/Emissivity Map (from L2B HEM product)
        """
        time_steps = []
        num_steps = max(len(l1b_paths), len(ctp_paths), len(hem_paths), 6)
        
        for i in range(num_steps):
            channels = []
            
            # Channel 0: WV (L1B)
            if i < len(l1b_paths):
                try:
                    with h5py.File(l1b_paths[i].replace('\\', '/'), 'r') as f:
                        wv = np.squeeze(f['IMG_WV'][:]).astype(np.float32)
                        wv = wv[::4, ::4]  # Fast subsample
                        channels.append(torch.from_numpy(np.nan_to_num(wv, nan=0.0)))
                except Exception:
                    channels.append(torch.zeros((352, 351)))
            else:
                channels.append(torch.zeros((352, 351)))
                
            # Channel 1: CTT (L2B CTP)
            if i < len(ctp_paths):
                try:
                    with h5py.File(ctp_paths[i].replace('\\', '/'), 'r') as f:
                        ctt = np.squeeze(f['CTT'][:]).astype(np.float32)
                        ctt = np.ma.filled(np.ma.masked_where(ctt < 0, ctt), 0.0)
                        channels.append(torch.from_numpy(ctt))
                except Exception:
                    channels.append(torch.zeros((313, 312)))
            else:
                channels.append(torch.zeros((313, 312)))
                
            # Channel 2: HEM
            if i < len(hem_paths):
                try:
                    with h5py.File(hem_paths[i].replace('\\', '/'), 'r') as f:
                        hem = np.squeeze(f['HEM'][:]).astype(np.float32)
                        hem = hem[::8, ::8]  # Fast subsample
                        channels.append(torch.from_numpy(np.nan_to_num(hem, nan=0.0)))
                except Exception:
                    channels.append(torch.zeros((352, 351)))
            else:
                channels.append(torch.zeros((352, 351)))
                
            # Resize individually (different raw sizes per channel)
            resized_channels = [resize_tensor(c, self.grid_size) for c in channels]
            time_steps.append(torch.stack(resized_channels, dim=0))
            
        # (Time=6, C=3, H, W) → (C=3, Time=6, H, W)
        insat_tensor = torch.stack(time_steps, dim=0).permute(1, 0, 2, 3)
        
        # Normalize channel-wise
        if self.normalize:
            insat_tensor = (insat_tensor - INSAT_MEAN) / (INSAT_STD + 1e-8)
        
        return insat_tensor

    def _load_targets(self, target_dict):
        """Load 3 binary target maps into (Channels=3, H, W).
        
        Channel order: [Cloudburst, Thunderstorm, FlashFlood]
        """
        cb = np.load(target_dict['cloudburst'].replace('\\', '/'))
        ts = np.load(target_dict['thunderstorm'].replace('\\', '/'))
        ff = np.load(target_dict['flash_flood'].replace('\\', '/'))
        
        cb = torch.from_numpy(cb.squeeze()).float()
        ts = torch.from_numpy(ts.squeeze()).float()
        ff = torch.from_numpy(ff.squeeze()).float()
        
        # Ensure 2D
        if len(cb.shape) > 2: cb = cb[0]
        if len(ts.shape) > 2: ts = ts[0]
        if len(ff.shape) > 2: ff = ff[0]
        
        cb = resize_tensor(cb, self.grid_size)
        ts = resize_tensor(ts, self.grid_size)
        ff = resize_tensor(ff, self.grid_size)
        
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
