"""
SIH26077 — Hybrid TransUNet: Spatiotemporal Multi-Task Model
==============================================================
A multi-task deep learning model for simultaneous prediction of:
  1. Cloudburst probability
  2. Severe Thunderstorm probability
  3. Flash Flood probability

Architecture (Hybrid CNN-Transformer):
  - Temporal Compression: 3D convolutions collapse 6 timesteps -> 1 feature map
  - Early Fusion: IMDAA (64ch) + INSAT (32ch) + Terrain (16ch) = 112 channels
  - Shared Encoder: 2-level U-Net backbone (128 -> 256 -> 512 bottleneck)
  - Transformer Bottleneck: Multi-Head Self-Attention (8 heads) with positional
    encoding at the bottleneck (64x64 spatial resolution). This provides global
    receptive field awareness, allowing the model to cross-reference distant
    atmospheric patterns (e.g., moisture from Arabian Sea triggering storms in
    the Himalayas) — satisfying the MoES requirement for "attention mechanisms".
  - Multi-Task Decoders: 3 independent U-Net decoders with skip connections

Input shapes:
  - IMDAA:   (B, 30, 6, 256, 256)  — 30 atmospheric channels, 6 timesteps
  - INSAT:   (B, 4, 6, 256, 256)   — 4 satellite channels (WV, CTT, HEM, CTT_RATE), 6 timesteps
  - Terrain: (B, 2, 256, 256)      — elevation + slope (static)

Output:
  - (B, 3, 256, 256) — logits for [Cloudburst, Thunderstorm, FlashFlood]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock2D(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU.
    
    Args:
        dropout: If > 0, adds Dropout2d after the second ReLU.
                 Use in decoder heads (not encoder) to prevent overfitting
                 on small datasets (< 100 training samples).
    """

    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class SelfAttention(nn.Module):
    """Multi-Head Self-Attention block for the TransUNet bottleneck.
    
    Flattens spatial dimensions (H*W) into a sequence of tokens,
    applies standard Multi-Head Attention with learned positional
    encoding, then reshapes back to 2D spatial feature maps.
    
    This enables the model to capture long-range spatial dependencies
    (global receptive field) that pure convolutions cannot achieve,
    while keeping compute manageable at 64x64 resolution (4096 tokens).
    
    Args:
        embed_dim: Number of channels in the feature map (default: 512)
        num_heads: Number of attention heads (default: 8)
        spatial_size: Spatial resolution at bottleneck (default: 64)
        dropout: Dropout rate for attention weights (default: 0.1)
    """
    
    def __init__(self, embed_dim=512, num_heads=8, spatial_size=64, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        
        # Layer normalization (pre-norm architecture for stability)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Standard Multi-Head Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Learned 2D positional encoding
        num_tokens = spatial_size * spatial_size  # 64*64 = 4096
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim) * 0.02
        )
        
        # Feed-forward network after attention (standard Transformer block)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) — feature map from the CNN encoder
        Returns:
            (B, C, H, W) — attention-enhanced feature map (same shape)
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C) where N = H*W
        
        # Add positional encoding
        x_pos = x_flat + self.pos_embed[:, :H*W, :]
        
        # Pre-norm + Multi-Head Self Attention + Residual
        x_norm = self.norm(x_pos)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out  # Residual connection
        
        # Feed-forward + Residual
        x_flat = x_flat + self.ffn(x_flat)
        
        # Reshape back to 2D: (B, N, C) -> (B, C, H, W)
        out = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return out


class SpatiotemporalMultiTaskModel(nn.Module):
    """Hybrid TransUNet: Multi-task spatiotemporal model for severe weather nowcasting.
    
    Combines a CNN encoder/decoder with a Transformer attention mechanism
    at the bottleneck for global atmospheric pattern recognition.
    """

    def __init__(self):
        super().__init__()
        
        # -- 1. Temporal Compression (3D -> 2D) ----------------------
        
        # IMDAA: (B, 30, T=6, H, W) -> (B, 64, 1, H, W) -> squeeze -> (B, 64, H, W)
        self.imdaa_time_compress = nn.Sequential(
            nn.Conv3d(30, 32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),  # T: 6->4
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(4, 3, 3), padding=(0, 1, 1)),  # T: 4->1
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        # INSAT: (B, 4, T=6, H, W) -> (B, 32, 1, H, W) -> squeeze -> (B, 32, H, W)
        # 4 channels: WV, CTT, HEM, CTT_RATE (drop rate, ps.md requirement)
        self.insat_time_compress = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=(3, 3, 3), padding=(0, 1, 1)),   # T: 6->4
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(4, 3, 3), padding=(0, 1, 1)),  # T: 4->1
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        # Terrain embedding: (B, 2, H, W) -> (B, 16, H, W)
        self.terrain_embed = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # -- 2. Shared Encoder ----------------------------------------
        # Fused channels: 64 (IMDAA) + 32 (INSAT) + 16 (Terrain) = 112
        
        self.enc1 = ConvBlock2D(112, 128)       # -> (B, 128, 256, 256)
        self.pool1 = nn.MaxPool2d(2)             # -> (B, 128, 128, 128)
        
        self.enc2 = ConvBlock2D(128, 256)        # -> (B, 256, 128, 128)
        self.pool2 = nn.MaxPool2d(2)             # -> (B, 256, 64, 64)
        
        # -- 3. Hybrid Bottleneck (CNN + Transformer) -----------------
        self.bottleneck = ConvBlock2D(256, 512)  # -> (B, 512, 64, 64)
        self.attention = SelfAttention(           # Transformer attention
            embed_dim=512,
            num_heads=8,
            spatial_size=64,
            dropout=0.1,
        )
        
        # -- 4. Multi-Task Decoder Heads ------------------------------
        # Each head has its own upsampling path for task-specific specialization
        
        # Head 1: Thunderstorm
        self.ts_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ts_dec1 = ConvBlock2D(256 + 256, 256, dropout=0.3)  # skip from enc2
        self.ts_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ts_dec2 = ConvBlock2D(128 + 128, 128, dropout=0.3)  # skip from enc1
        self.ts_out = nn.Conv2d(128, 1, kernel_size=1)
        
        # Head 2: Cloudburst
        self.cb_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.cb_dec1 = ConvBlock2D(256 + 256, 256, dropout=0.3)
        self.cb_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.cb_dec2 = ConvBlock2D(128 + 128, 128, dropout=0.3)
        self.cb_out = nn.Conv2d(128, 1, kernel_size=1)
        
        # Head 3: Flash Flood
        self.ff_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ff_dec1 = ConvBlock2D(256 + 256, 256, dropout=0.3)
        self.ff_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ff_dec2 = ConvBlock2D(128 + 128, 128, dropout=0.3)
        self.ff_out = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, imdaa, insat, terrain):
        """
        Args:
            imdaa:   (B, 30, 6, 256, 256) -- IMDAA reanalysis
            insat:   (B, 3, 6, 256, 256)  -- INSAT satellite
            terrain: (B, 2, 256, 256)     -- DEM terrain
            
        Returns:
            (B, 3, 256, 256) -- logits for [Cloudburst, Thunderstorm, FlashFlood]
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
        
        # 4. Hybrid bottleneck: CNN + Transformer Self-Attention
        b = self.bottleneck(x_pool2)  # (B, 512, 64, 64) — CNN features
        b = self.attention(b)          # (B, 512, 64, 64) — attention-enhanced
        
        # 5. Multi-task decoders
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
    print("Testing Hybrid TransUNet (SpatiotemporalMultiTaskModel)...")
    
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
    
    # Count attention-specific parameters
    attn_params = sum(p.numel() for p in model.attention.parameters())
    print(f"Attention parameters: {attn_params:,} ({100*attn_params/total_params:.1f}% of total)")
    
    # Test forward pass
    preds = model(imdaa_dummy, insat_dummy, terrain_dummy)
    
    print(f"Model output shape: {preds.shape} (Expected: 2, 3, 256, 256)")
    print("[OK] Hybrid TransUNet model definition passed.")
