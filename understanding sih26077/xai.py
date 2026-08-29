"""
SIH26077 — Explainable AI (XAI) Module
=========================================
GradCAM-based explainability for the Multi-Task U-Net.

Generates spatial heatmaps showing which regions of the input the model
focused on when making each prediction. This satisfies the PS requirement
for "transparently displaying meteorological triggers."

Usage:
    from xai import GradCAM
    
    cam = GradCAM(model, target_layer='bottleneck')
    heatmap = cam.generate(imdaa, insat, terrain, target_class=0)  # Cloudburst
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.cm as cm
from model import SpatiotemporalMultiTaskModel

from config import DEVICE, CHECKPOINT_PATH, TARGET_NAMES

import os
import warnings
warnings.filterwarnings('ignore')


class GradCAM:
    """Gradient-weighted Class Activation Mapping for the multi-task U-Net.
    
    Hooks into a target layer (default: bottleneck) and computes the
    gradient of a specific output class with respect to that layer's
    activations. The resulting heatmap highlights spatial regions that
    were most influential for the prediction.
    
    Args:
        model: A SpatiotemporalMultiTaskModel instance (on device)
        target_layer: Name of the layer to hook ('bottleneck', 'enc1', 'enc2')
    """

    def __init__(self, model, target_layer='bottleneck'):
        self.model = model
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        # Get the target layer
        layer = getattr(model, target_layer)
        
        # Register hooks
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, imdaa, insat, terrain, target_class=0):
        """Generate a GradCAM heatmap for a specific risk class.
        
        Args:
            imdaa:   (1, 30, 6, H, W) tensor on device
            insat:   (1, 3, 6, H, W)  tensor on device
            terrain: (1, 2, H, W)     tensor on device
            target_class: 0=Cloudburst, 1=Thunderstorm, 2=FlashFlood
            
        Returns:
            heatmap: (H, W) numpy array in [0, 1], same spatial size as input
        """
        self.model.zero_grad()
        
        # Forward pass (need gradients for CAM)
        imdaa.requires_grad_(True)
        output = self.model(imdaa, insat, terrain)  # (1, 3, H, W)
        
        # Select the target class and compute mean activation
        target_output = output[0, target_class].mean()
        
        # Backward pass
        target_output.backward(retain_graph=True)
        
        # Compute GradCAM
        # Global average pool the gradients → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # Only positive contributions
        
        # Upsample to input spatial size
        cam = F.interpolate(cam, size=imdaa.shape[-2:], mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().float().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

    def generate_rgba_overlay(self, imdaa, insat, terrain, target_class=0, 
                               colormap='jet', alpha_scale=0.7):
        """Generate an RGBA overlay image suitable for Folium map display.
        
        Args:
            target_class: 0=Cloudburst, 1=Thunderstorm, 2=FlashFlood
            colormap: Matplotlib colormap name
            alpha_scale: Maximum alpha for the overlay
            
        Returns:
            rgba_image: (H, W, 4) numpy array with alpha channel
        """
        heatmap = self.generate(imdaa, insat, terrain, target_class)
        
        # Apply colormap
        cmap = matplotlib.colormaps[colormap]
        rgba_image = cmap(heatmap)
        rgba_image[..., 3] = np.clip(heatmap * alpha_scale, 0.0, alpha_scale)
        
        return rgba_image


def load_xai_model():
    """Load a model with GradCAM ready for explanation generation."""
    model = SpatiotemporalMultiTaskModel().to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        )
    
    cam = GradCAM(model, target_layer='bottleneck')
    return model, cam


# ============================================================
# Self-Test
# ============================================================
if __name__ == "__main__":
    print("Testing GradCAM XAI Module...")
    
    model = SpatiotemporalMultiTaskModel().to(DEVICE)
    cam = GradCAM(model, target_layer='bottleneck')
    
    # Dummy input
    imdaa = torch.randn(1, 30, 6, 256, 256, device=DEVICE)
    insat = torch.randn(1, 3, 6, 256, 256, device=DEVICE)
    terrain = torch.randn(1, 2, 256, 256, device=DEVICE)
    
    for i, name in enumerate(TARGET_NAMES):
        heatmap = cam.generate(imdaa, insat, terrain, target_class=i)
        print(f"  {name} GradCAM: shape={heatmap.shape}, "
              f"min={heatmap.min():.3f}, max={heatmap.max():.3f}")
    
    # Test RGBA overlay
    rgba = cam.generate_rgba_overlay(imdaa, insat, terrain, target_class=0)
    print(f"  RGBA overlay shape: {rgba.shape}")
    
    print("[OK] XAI module test passed.")
