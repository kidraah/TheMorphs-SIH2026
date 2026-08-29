"""
SIH26077 — Interactive Web Dashboard
======================================
Streamlit + Folium web-based spatial dashboard for disaster management.

Features:
  - Dynamic risk map overlaid on interactive map (Himachal/Uttarakhand)
  - GradCAM Explainable AI visualization
  - Real-time automated alert panel
  - Configurable lead time and time window selection
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import torch
import numpy as np
import matplotlib
import matplotlib.cm as cm
import os

from model import SpatiotemporalMultiTaskModel
from data_loader import SpatiotemporalDataset
from xai import GradCAM
from alerts import AlertEngine

from config import (
    DEVICE, CHECKPOINT_PATH, TARGET_NAMES, LEAD_TIMES,
    DEFAULT_LEAD_TIME, MAP_BOUNDS, MAP_CENTER,
)

st.set_page_config(
    page_title="SIH26077 — AI Early Warning System",
    page_icon="⛈️",
    layout="wide",
)


@st.cache_resource
def load_backend(lead_time):
    """Load the model, dataset, and XAI engine into memory once."""
    print(f"Loading PyTorch Backend (lead_time={lead_time}h)...")
    
    model = SpatiotemporalMultiTaskModel().to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        )
    model.eval()
    
    dataset = SpatiotemporalDataset(lead_time=lead_time)
    cam = GradCAM(model, target_layer='bottleneck')
    alert_engine = AlertEngine()
    
    return model, dataset, cam, alert_engine


@st.cache_resource
def get_dataset_sample(_dataset, lead_time, window_index):
    """Cache the disk I/O so we don't read 180 files every UI click."""
    return _dataset[window_index]


def run_inference(model, dataset, lead_time, window_index):
    """Run model inference and return probability maps for all 3 risk types."""
    sample = get_dataset_sample(dataset, lead_time, window_index)
    
    imdaa = sample['imdaa'].unsqueeze(0).to(DEVICE)
    insat = sample['insat'].unsqueeze(0).to(DEVICE)
    terrain = sample['terrain'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            preds = model(imdaa, insat, terrain)
        probs = torch.sigmoid(preds)[0]  # (3, H, W)
        
    return {
        'cloudburst': probs[0].float().cpu().numpy(),
        'thunderstorm': probs[1].float().cpu().numpy(),
        'flash_flood': probs[2].float().cpu().numpy(),
    }, imdaa, insat, terrain


def prob_to_rgba(risk_matrix, colormap='jet', alpha_scale=0.7):
    """Convert a probability matrix to an RGBA image for map overlay."""
    cmap = matplotlib.colormaps[colormap]
    rgba_image = cmap(risk_matrix)
    rgba_image[..., 3] = np.clip(risk_matrix * 1.5, 0.0, alpha_scale)
    return rgba_image


# ── UI Layout ──────────────────────────────────────────────────

st.title("⛈️ SIH26077 — Hyper-Local Early Warning System")
st.markdown(
    "An AI Spatiotemporal Predictive Engine for Severe Weather Nowcasting "
    "in the Himalayas — *Ministry of Earth Sciences (MoES)*"
)

# ── Sidebar Controls ──────────────────────────────────────────

st.sidebar.header("🎛️ Control Panel")

# Lead time selector
lead_time = st.sidebar.select_slider(
    "Prediction Lead Time (hours)",
    options=LEAD_TIMES,
    value=DEFAULT_LEAD_TIME,
)

# Load backend
with st.spinner("Loading PyTorch Engine & Geospatial Data..."):
    model, dataset, cam, alert_engine = load_backend(lead_time)

# Window selector
window_options = [
    f"Window {i} ({dataset.windows[i].get('win_id', f'win_{i}')})"
    for i in range(len(dataset))
]
selected_window_str = st.sidebar.selectbox(
    "Select Time Window",
    window_options,
    index=min(17, len(window_options) - 1),
)
selected_window_idx = window_options.index(selected_window_str)

# Risk type selector
risk_type = st.sidebar.radio(
    "Risk Map to Display",
    ("Cloudburst", "Severe Thunderstorm", "Flash Flood"),
)

# XAI toggle
show_xai = st.sidebar.checkbox("Show XAI Explanation (GradCAM)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Region:** Himachal Pradesh & Uttarakhand")
st.sidebar.markdown(f"**Device:** `{DEVICE}`")

# ── Run Inference ─────────────────────────────────────────────

with st.spinner(f"Running Neural Inference for {risk_type}..."):
    risk_maps, imdaa_t, insat_t, terrain_t = run_inference(
        model, dataset, lead_time, selected_window_idx
    )

# Map risk type key
risk_key_map = {
    "Cloudburst": "cloudburst",
    "Severe Thunderstorm": "thunderstorm",
    "Flash Flood": "flash_flood",
}
risk_key = risk_key_map[risk_type]
target_idx = list(risk_key_map.keys()).index(risk_type)

# ── Map Display ───────────────────────────────────────────────

col_map, col_alerts = st.columns([3, 1])

with col_map:
    st.subheader(f"🗺️ {risk_type} Risk Map")
    
    m = folium.Map(location=MAP_CENTER, zoom_start=7, tiles="OpenStreetMap")
    
    # Risk map overlay
    rgba_overlay = prob_to_rgba(risk_maps[risk_key])
    folium.raster_layers.ImageOverlay(
        image=rgba_overlay,
        bounds=MAP_BOUNDS,
        opacity=1.0,
        name=f"{risk_type} Risk",
    ).add_to(m)
    
    # XAI overlay (optional)
    if show_xai:
        with st.spinner("Generating GradCAM explanation..."):
            xai_rgba = cam.generate_rgba_overlay(
                imdaa_t, insat_t, terrain_t,
                target_class=target_idx,
                colormap='hot',
                alpha_scale=0.5,
            )
        folium.raster_layers.ImageOverlay(
            image=xai_rgba,
            bounds=MAP_BOUNDS,
            opacity=1.0,
            name="XAI: Model Attention",
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    st_folium(m, width=900, height=600)

# ── Alert Panel ───────────────────────────────────────────────

with col_alerts:
    st.subheader("🚨 Alert Panel")
    
    result = alert_engine.evaluate(risk_maps)
    
    # Severity badge
    severity = result['highest_severity']
    severity_colors = {
        'EMERGENCY': '🔴',
        'WARNING': '🟠',
        'WATCH': '🟡',
        'NONE': '🟢',
    }
    st.markdown(f"### {severity_colors.get(severity, '⚪')} Status: **{severity}**")
    st.caption(result['summary'])
    
    st.markdown("---")
    
    if result['alerts']:
        for alert in result['alerts']:
            with st.expander(
                f"{severity_colors.get(alert['severity'], '📢')} "
                f"{alert['severity']}: {alert['risk_type']}",
                expanded=(alert['severity'] == 'EMERGENCY'),
            ):
                st.metric("Max Probability", f"{alert['max_probability']:.1%}")
                st.metric("Affected Area", f"{alert['affected_area_pct']:.1f}%")
                st.caption(alert['message'])
    else:
        st.success("All clear — no alerts at this time.")

# ── XAI Explanation Panel ─────────────────────────────────────

if show_xai:
    st.markdown("---")
    st.subheader("🔍 Explainable AI — Model Attention Map")
    st.markdown(
        f"The heatmap above (toggle 'XAI: Model Attention' layer) shows which "
        f"spatial regions the model focused on most when predicting **{risk_type}** risk. "
        f"Brighter areas = higher model attention = stronger meteorological triggers detected."
    )

# Footer
st.markdown("---")
st.caption(
    "SIH26077 — AI-Driven Hyper-Local Early Warning System | "
    "Ministry of Earth Sciences (MoES) | Smart India Hackathon 2026"
)
