"""
SIH26077 — Spatiotemporal Multi-Task U-Net
============================================
A multi-task deep learning model for simultaneous prediction of:
  1. Cloudburst probability
  2. Severe Thunderstorm probability
  3. Flash Flood probability

Architecture:
  - Temporal Compression: 3D convolutions collapse 6 timesteps → 1 feature map
  - Early Fusion: IMDAA (64ch) + INSAT (32ch) + Terrain (16ch) = 112 channels
  - Shared Encoder: 2-level U-Net backbone (128 → 256 → 512 bottleneck)
  - Multi-Task Decoders: 3 independent U-Net decoders with skip connections

Input shapes:
  - IMDAA:   (B, 30, 6, 256, 256)  — 30 atmospheric channels, 6 timesteps
  - INSAT:   (B, 3, 6, 256, 256)   — 3 satellite channels, 6 timesteps
  - Terrain: (B, 2, 256, 256)      — elevation + slope (static)

Output:
  - (B, 3, 256, 256) — logits for [Cloudburst, Thunderstorm, FlashFlood]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SpatiotemporalMultiTaskModel(nn.Module):
    """Multi-task spatiotemporal U-Net for severe weather nowcasting."""

    def __init__(self):
        super().__init__()
        
        # ── 1. Temporal Compression (3D → 2D) ──────────────────
        
        # IMDAA: (B, 30, T=6, H, W) → (B, 64, 1, H, W) → squeeze → (B, 64, H, W)
        self.imdaa_time_compress = nn.Sequential(
            nn.Conv3d(30, 32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),  # T: 6→4
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(4, 3, 3), padding=(0, 1, 1)),  # T: 4→1
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        # INSAT: (B, 3, T=6, H, W) → (B, 32, 1, H, W) → squeeze → (B, 32, H, W)
        self.insat_time_compress = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(0, 1, 1)),   # T: 6→4
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(4, 3, 3), padding=(0, 1, 1)),  # T: 4→1
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        # Terrain embedding: (B, 2, H, W) → (B, 16, H, W)
        self.terrain_embed = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # ── 2. Shared Encoder ───────────────────────────────────
        # Fused channels: 64 (IMDAA) + 32 (INSAT) + 16 (Terrain) = 112
        
        self.enc1 = ConvBlock2D(112, 128)       # → (B, 128, 256, 256)
        self.pool1 = nn.MaxPool2d(2)             # → (B, 128, 128, 128)
        
        self.enc2 = ConvBlock2D(128, 256)        # → (B, 256, 128, 128)
        self.pool2 = nn.MaxPool2d(2)             # → (B, 256, 64, 64)
        
        self.bottleneck = ConvBlock2D(256, 512)  # → (B, 512, 64, 64)
        
        # ── 3. Multi-Task Decoder Heads ─────────────────────────
        # Each head has its own upsampling path for task-specific specialization
        
        # Head 1: Thunderstorm
        self.ts_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ts_dec1 = ConvBlock2D(256 + 256, 256)  # skip from enc2
        self.ts_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ts_dec2 = ConvBlock2D(128 + 128, 128)  # skip from enc1
        self.ts_out = nn.Conv2d(128, 1, kernel_size=1)
        
        # Head 2: Cloudburst
        self.cb_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.cb_dec1 = ConvBlock2D(256 + 256, 256)
        self.cb_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.cb_dec2 = ConvBlock2D(128 + 128, 128)
        self.cb_out = nn.Conv2d(128, 1, kernel_size=1)
        
        # Head 3: Flash Flood
        self.ff_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ff_dec1 = ConvBlock2D(256 + 256, 256)
        self.ff_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ff_dec2 = ConvBlock2D(128 + 128, 128)
        self.ff_out = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, imdaa, insat, terrain):
        """
        Args:
            imdaa:   (B, 30, 6, 256, 256) — IMDAA reanalysis
            insat:   (B, 3, 6, 256, 256)  — INSAT satellite
            terrain: (B, 2, 256, 256)     — DEM terrain
            
        Returns:
            (B, 3, 256, 256) — logits for [Cloudburst, Thunderstorm, FlashFlood]
        """
        # 1. Temporal compression
        x_imdaa = self.imdaa_time_compress(imdaa).squeeze(2)    # (B, 64, H, W)
        x_insat = self.insat_time_compress(insat).squeeze(2)     # (B, 32, H, W)
        x_terrain = self.terrain_embed(terrain)                   # (B, 16, H, W)
        
        # 2. Early fusion (concatenation)
        x = torch.cat([x_imdaa, x_insat, x_terrain], dim=1)     # (B, 112, H, W)
        
        # 3. Shared encoder
        e1 = self.enc1(x)             # (B, 128, 256, 256)
        x_pool1 = self.pool1(e1)      # (B, 128, 128, 128)
        
        e2 = self.enc2(x_pool1)       # (B, 256, 128, 128)
        x_pool2 = self.pool2(e2)      # (B, 256, 64, 64)
        
        b = self.bottleneck(x_pool2)  # (B, 512, 64, 64)
        
        # 4. Multi-task decoders
        # Thunderstorm head
        ts_u1 = self.ts_up1(b)
        ts_d1 = self.ts_dec1(torch.cat([ts_u1, e2], dim=1))
        ts_u2 = self.ts_up2(ts_d1)
        ts_d2 = self.ts_dec2(torch.cat([ts_u2, e1], dim=1))
        ts_pred = self.ts_out(ts_d2)   # (B, 1, 256, 256)
        
        # Cloudburst head
        cb_u1 = self.cb_up1(b)
        cb_d1 = self.cb_dec1(torch.cat([cb_u1, e2], dim=1))
        cb_u2 = self.cb_up2(cb_d1)
        cb_d2 = self.cb_dec2(torch.cat([cb_u2, e1], dim=1))
        cb_pred = self.cb_out(cb_d2)   # (B, 1, 256, 256)
        
        # Flash Flood head
        ff_u1 = self.ff_up1(b)
        ff_d1 = self.ff_dec1(torch.cat([ff_u1, e2], dim=1))
        ff_u2 = self.ff_up2(ff_d1)
        ff_d2 = self.ff_dec2(torch.cat([ff_u2, e1], dim=1))
        ff_pred = self.ff_out(ff_d2)   # (B, 1, 256, 256)
        
        # Output: [Cloudburst, Thunderstorm, FlashFlood] — matches target order
        return torch.cat([cb_pred, ts_pred, ff_pred], dim=1)


# ============================================================
# Self-Test
# ============================================================
if __name__ == "__main__":
    print("Testing SpatiotemporalMultiTaskModel...")
    
    # Dummy tensors simulating batch size 2
    imdaa_dummy = torch.randn(2, 30, 6, 256, 256)
    insat_dummy = torch.randn(2, 3, 6, 256, 256)
    terrain_dummy = torch.randn(2, 2, 256, 256)
    
    model = SpatiotemporalMultiTaskModel()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    preds = model(imdaa_dummy, insat_dummy, terrain_dummy)
    
    print(f"Model output shape: {preds.shape} (Expected: 2, 3, 256, 256)")
    print("[OK] Model definition passed.")
