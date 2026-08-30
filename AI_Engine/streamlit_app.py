"""
SIH26077 — Streamlit Risk Map Viewer
======================================
Interactive animated risk map for the Aug 15–25 2019 Uttarakhand/HP cloudbursts.
Runs your trained Hybrid TransUNet on every time window and renders:
  • Folium map with real AI probability heatmaps (Cloudburst, Thunderstorm, Flash Flood)
  • District risk choropleth (color-coded by max model probability)
  • Per-district risk table
  • Model confidence metrics

Run with:
    streamlit run streamlit_app.py
"""

import io
import json
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta, timezone

matplotlib.use("Agg")

# ── Paths & config ───────────────────────────────────────────────────────────
from config import (
    DEVICE, CHECKPOINT_PATH, INDEX_PATH, LEAD_TIMES,
    DEFAULT_LEAD_TIME, MAP_BOUNDS, MAP_CENTER,
)
from model import SpatiotemporalMultiTaskModel
from data_loader import SpatiotemporalDataset

IST = timezone(timedelta(hours=5, minutes=30))

LAT_MIN, LAT_MAX = MAP_BOUNDS[0][0], MAP_BOUNDS[1][0]
LON_MIN, LON_MAX = MAP_BOUNDS[0][1], MAP_BOUNDS[1][1]

# ── District coordinates (lat, lon, display_name, state) ────────────────────
DISTRICTS = {
    "uttarkashi":       (30.73, 78.44, "Uttarkashi",       "Uttarakhand"),
    "chamoli":          (30.42, 79.33, "Chamoli",           "Uttarakhand"),
    "rudraprayag":      (30.28, 79.00, "Rudraprayag",       "Uttarakhand"),
    "tehri-garhwal":    (30.39, 78.48, "Tehri Garhwal",     "Uttarakhand"),
    "bageshwar":        (29.84, 79.77, "Bageshwar",         "Uttarakhand"),
    "pithoragarh":      (29.58, 80.22, "Pithoragarh",       "Uttarakhand"),
    "pauri-garhwal":    (30.15, 78.78, "Pauri Garhwal",     "Uttarakhand"),
    "almora":           (29.60, 79.66, "Almora",            "Uttarakhand"),
    "nainital":         (29.38, 79.46, "Nainital",          "Uttarakhand"),
    "champawat":        (29.33, 80.09, "Champawat",         "Uttarakhand"),
    "dehradun":         (30.32, 78.03, "Dehradun",          "Uttarakhand"),
    "haridwar":         (29.88, 78.16, "Haridwar",          "Uttarakhand"),
    "kullu":            (31.96, 77.11, "Kullu",             "Himachal Pradesh"),
    "mandi":            (31.71, 76.93, "Mandi",             "Himachal Pradesh"),
    "chamba":           (32.56, 76.13, "Chamba",            "Himachal Pradesh"),
    "shimla":           (31.10, 77.17, "Shimla",            "Himachal Pradesh"),
    "kangra":           (32.10, 76.27, "Kangra",            "Himachal Pradesh"),
    "sirmaur":          (30.56, 77.67, "Sirmaur",           "Himachal Pradesh"),
    "kinnaur":          (31.58, 78.17, "Kinnaur",           "Himachal Pradesh"),
    "solan":            (30.90, 77.10, "Solan",             "Himachal Pradesh"),
    "bilaspur":         (31.33, 76.76, "Bilaspur",          "Himachal Pradesh"),
    "hamirpur":         (31.68, 76.52, "Hamirpur",          "Himachal Pradesh"),
    "una":              (31.47, 76.27, "Una",               "Himachal Pradesh"),
}

RISK_COLORS = {
    "EXTREME":   "#9333EA",
    "VERY HIGH": "#EF4444",
    "HIGH":      "#F97316",
    "MODERATE":  "#EAB308",
    "LOW":       "#22C55E",
}

# ── Model & dataset caching ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Hybrid TransUNet model...")
def load_model():
    model = SpatiotemporalMultiTaskModel()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@st.cache_resource(show_spinner="Loading dataset windows...")
def load_dataset(lead: str):
    return SpatiotemporalDataset(lead_time=lead)


@st.cache_resource(show_spinner="Reading window index...")
def load_window_index():
    with open(INDEX_PATH) as f:
        return json.load(f)

# ── Inference ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=50)
def run_inference(window_idx: int, lead: str):
    """Run the model on one window. Cached so replaying doesn't re-infer."""
    model   = load_model()
    dataset = load_dataset(lead)

    if window_idx >= len(dataset):
        st.error(f"Window {window_idx} out of range (max {len(dataset)-1})")
        return None

    sample = dataset[window_idx]
    imdaa   = sample["imdaa"].unsqueeze(0).to(DEVICE)
    insat   = sample["insat"].unsqueeze(0).to(DEVICE)
    terrain = sample["terrain"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.autocast(device_type=DEVICE.type if hasattr(DEVICE, "type") else "cpu",
                            dtype=torch.bfloat16, enabled=(str(DEVICE) == "cuda")):
            preds = model(imdaa, insat, terrain)
        probs = torch.sigmoid(preds)[0].float().cpu().numpy()  # (3, H, W)

    return {
        "cloudburst":    probs[0],
        "thunderstorm":  probs[1],
        "flash_flood":   probs[2],
    }


def latlon_to_grid(lat, lon, h=256, w=256):
    r = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * h)
    c = int((lon   - LON_MIN) / (LON_MAX - LON_MIN) * w)
    return max(0, min(r, h-1)), max(0, min(c, w-1))


def district_prob(prob_map, lat, lon, patch=6):
    r, c = latlon_to_grid(lat, lon)
    r0, r1 = max(0, r-patch), min(255, r+patch)
    c0, c1 = max(0, c-patch), min(255, c+patch)
    return float(prob_map[r0:r1, c0:c1].mean())


def prob_to_risk(p):
    if p >= 0.70: return "EXTREME"
    if p >= 0.55: return "VERY HIGH"
    if p >= 0.40: return "HIGH"
    if p >= 0.25: return "MODERATE"
    return "LOW"


def smooth_prob_to_heatmap_points(prob_map, n_samples=2000, threshold=0.12):
    """Convert a 256×256 probability map to weighted (lat, lon, weight) points for folium HeatMap."""
    smoothed = np.clip(gaussian_filter(prob_map.astype(np.float64), sigma=3.0), 0.0, 1.0)
    points = []
    h, w = smoothed.shape
    ys, xs = np.where(smoothed > threshold)
    if len(ys) == 0:
        return []
    # Sample random subset weighted by probability
    weights = smoothed[ys, xs]
    total   = weights.sum()
    if total == 0:
        return []
    probs_norm = weights / total
    chosen = np.random.choice(len(ys), size=min(n_samples, len(ys)), replace=False, p=probs_norm)
    for idx in chosen:
        r, c = ys[idx], xs[idx]
        lat = LAT_MAX - (r / h) * (LAT_MAX - LAT_MIN)
        lon = LON_MIN + (c / w) * (LON_MAX - LON_MIN)
        points.append([lat, lon, float(weights[idx])])
    return points


def get_timestamp(win_meta, win_idx):
    raw = win_meta.get("window_end", win_meta.get("window_start", ""))
    try:
        dt = datetime.fromisoformat(raw) + timedelta(hours=5, minutes=30)
        return dt.strftime("%d %b %Y  %H:%M IST")
    except Exception:
        return f"Window {win_idx}"

# ── Build the Folium map ─────────────────────────────────────────────────────

def build_map(maps, active_layers, district_results, timestamp_label):
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=7,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
    )

    LAYER_CONFIG = {
        "cloudburst":   {"color": ["#ffffcc","#fd8d3c","#bd0026","#49006a"], "name": "🌧 Cloudburst Risk"},
        "thunderstorm": {"color": ["#f7fbff","#6baed6","#2171b5","#084594"], "name": "⚡ Thunderstorm Risk"},
        "flash_flood":  {"color": ["#f7fcfd","#41b6c4","#0c2c84","#67001f"], "name": "🌊 Flash Flood Risk"},
    }

    for layer_key, cfg in LAYER_CONFIG.items():
        if active_layers.get(layer_key, False) and maps and layer_key in maps:
            pts = smooth_prob_to_heatmap_points(maps[layer_key], n_samples=3000)
            if pts:
                fg = folium.FeatureGroup(name=cfg["name"], show=True)
                HeatMap(
                    pts,
                    name=cfg["name"],
                    min_opacity=0.2,
                    max_opacity=0.85,
                    radius=22,
                    blur=18,
                    gradient={str(round(i * 0.25, 2)): c for i, c in enumerate(cfg["color"])},
                ).add_to(fg)
                fg.add_to(m)

    # District markers
    for dist_id, (lat, lon, name, state) in DISTRICTS.items():
        result = district_results.get(dist_id, {})
        risk   = result.get("risk", "LOW")
        cb_p   = result.get("cloudburst", 0.0)
        ts_p   = result.get("thunderstorm", 0.0)
        ff_p   = result.get("flash_flood", 0.0)
        color  = RISK_COLORS.get(risk, "#22C55E")

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:180px">
          <b style="font-size:14px">{name}</b>
          <div style="color:#666;font-size:11px">{state}</div>
          <div style="margin-top:6px;padding:3px 8px;border-radius:4px;background:{color};color:white;
               font-weight:bold;text-align:center;font-size:12px">{risk}</div>
          <table style="width:100%;margin-top:6px;font-size:11px;border-collapse:collapse">
            <tr><td>Cloudburst</td><td align=right><b>{cb_p*100:.1f}%</b></td></tr>
            <tr><td>Thunderstorm</td><td align=right><b>{ts_p*100:.1f}%</b></td></tr>
            <tr><td>Flash Flood</td><td align=right><b>{ff_p*100:.1f}%</b></td></tr>
          </table>
          <div style="color:#999;font-size:9px;margin-top:4px">{timestamp_label}</div>
        </div>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=8 if risk in ("EXTREME", "VERY HIGH") else 6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{name} — {risk}",
        ).add_to(m)

    # Timestamp watermark
    ts_html = f"""
    <div style="position:fixed;bottom:30px;left:50%;transform:translateX(-50%);
         background:rgba(10,56,113,0.88);color:white;padding:6px 18px;
         border-radius:20px;font-family:monospace;font-size:13px;font-weight:bold;
         z-index:9999;letter-spacing:1px;pointer-events:none">
      ⏱ {timestamp_label}
    </div>
    """
    m.get_root().html.add_child(folium.Element(ts_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m

# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SIH26077 · AI Flood Risk Viewer",
    page_icon="🌧",
    layout="wide",
)

# Header
st.markdown("""
<div style="background:linear-gradient(90deg,#0A3871,#1D5FAD);padding:16px 24px;border-radius:10px;margin-bottom:16px">
  <h2 style="color:white;margin:0;font-family:sans-serif">
    🛰 SIH26077 — AI-Driven Hyper-Local Early Warning System
  </h2>
  <p style="color:#B3D4F5;margin:4px 0 0;font-size:13px">
    Hybrid TransUNet · Uttarakhand &amp; Himachal Pradesh · Aug 15–25, 2019 Cloudburst Events
  </p>
</div>
""", unsafe_allow_html=True)

# Load resources
try:
    window_index = load_window_index()
    n_windows    = len(window_index)
except Exception as e:
    st.error(f"Could not load window index: {e}")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    lead_time = st.selectbox(
        "Prediction Lead Time",
        options=["2", "3", "4", "5", "6"],
        index=1,
        format_func=lambda x: f"{x} hours ahead",
    )

    window_idx = st.slider(
        "Time Window",
        min_value=0,
        max_value=n_windows - 1,
        value=min(17, n_windows - 1),
        step=1,
        help="Window 17 = Uttarkashi cloudburst peak (Aug 18, 2019)",
    )

    win_meta  = window_index[window_idx] if window_idx < n_windows else {}
    ts_label  = get_timestamp(win_meta, window_idx)
    st.info(f"📅 **{ts_label}**")

    st.markdown("---")
    st.markdown("### 🗺 Overlay Layers")
    show_cloudburst   = st.checkbox("🌧 Cloudburst Heatmap",   value=True)
    show_thunderstorm = st.checkbox("⚡ Thunderstorm Heatmap", value=False)
    show_flash_flood  = st.checkbox("🌊 Flash Flood Heatmap",  value=False)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
**Model:** Hybrid TransUNet  
**Input:** IMDAA (26 channels) + INSAT-3D + SRTM-DEM  
**Output:** 256×256 probability maps  
**Events:** 25 windows across Aug 15–25, 2019  
**Validated on:** Aug 16–25 2019 (5-day holdout)
    """)

# ── Run inference ────────────────────────────────────────────────────────────
with st.spinner(f"Running AI inference on Window {window_idx} (Lead: {lead_time}h)..."):
    try:
        maps = run_inference(window_idx, lead_time)
    except Exception as e:
        st.error(f"Inference failed: {e}")
        maps = None

# ── Compute district risks ────────────────────────────────────────────────────
district_results = {}
if maps:
    for dist_id, (lat, lon, name, state) in DISTRICTS.items():
        cb_p = district_prob(maps["cloudburst"],   lat, lon)
        ts_p = district_prob(maps["thunderstorm"], lat, lon)
        ff_p = district_prob(maps["flash_flood"],  lat, lon)
        max_p = max(cb_p, ts_p, ff_p)
        district_results[dist_id] = {
            "name":         name,
            "state":        state,
            "risk":         prob_to_risk(max_p),
            "cloudburst":   cb_p,
            "thunderstorm": ts_p,
            "flash_flood":  ff_p,
            "max_prob":     max_p,
        }

# ── Summary KPIs ─────────────────────────────────────────────────────────────
if maps:
    cb_max = float(maps["cloudburst"].max())
    ts_max = float(maps["thunderstorm"].max())
    ff_max = float(maps["flash_flood"].max())
    n_extreme  = sum(1 for d in district_results.values() if d["risk"] == "EXTREME")
    n_veryhigh = sum(1 for d in district_results.values() if d["risk"] == "VERY HIGH")
    n_high     = sum(1 for d in district_results.values() if d["risk"] == "HIGH")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🌧 Cloudburst",    f"{cb_max*100:.1f}%", help="Peak model probability")
    col2.metric("⚡ Thunderstorm",  f"{ts_max*100:.1f}%", help="Peak model probability")
    col3.metric("🌊 Flash Flood",   f"{ff_max*100:.1f}%", help="Peak model probability")
    col4.metric("🔴 EXTREME Districts", n_extreme)
    col5.metric("🟠 HIGH+ Districts",   n_extreme + n_veryhigh + n_high)

# ── Map ───────────────────────────────────────────────────────────────────────
active_layers = {
    "cloudburst":   show_cloudburst,
    "thunderstorm": show_thunderstorm,
    "flash_flood":  show_flash_flood,
}

m = build_map(maps, active_layers, district_results, ts_label)
st_folium(m, width="100%", height=520, returned_objects=[])

# ── District Risk Table ───────────────────────────────────────────────────────
if district_results:
    st.markdown("### 📊 District Risk Breakdown")
    
    import pandas as pd
    rows = []
    for d in sorted(district_results.values(), key=lambda x: -x["max_prob"]):
        risk = d["risk"]
        rows.append({
            "District":         d["name"],
            "State":            d["state"],
            "Risk Level":       risk,
            "Cloudburst (%)":   f"{d['cloudburst']*100:.1f}",
            "Thunderstorm (%)": f"{d['thunderstorm']*100:.1f}",
            "Flash Flood (%)":  f"{d['flash_flood']*100:.1f}",
        })
    
    df = pd.DataFrame(rows)
    
    def color_risk(val):
        colors = {
            "EXTREME":   "background-color:#9333EA;color:white;font-weight:bold",
            "VERY HIGH": "background-color:#EF4444;color:white;font-weight:bold",
            "HIGH":      "background-color:#F97316;color:white;font-weight:bold",
            "MODERATE":  "background-color:#EAB308;color:black;font-weight:bold",
            "LOW":       "background-color:#22C55E;color:white;font-weight:bold",
        }
        return colors.get(val, "")
    
    styled = df.style.applymap(color_risk, subset=["Risk Level"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Raw probability images ─────────────────────────────────────────────────
if maps:
    with st.expander("🔬 Raw Model Output (256×256 probability tensors)"):
        c1, c2, c3 = st.columns(3)
        for col, key, title, cmap in [
            (c1, "cloudburst",   "Cloudburst",   "Reds"),
            (c2, "thunderstorm", "Thunderstorm", "Purples"),
            (c3, "flash_flood",  "Flash Flood",  "Blues"),
        ]:
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(maps[key], cmap=cmap, vmin=0, vmax=1, origin="upper",
                           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
            ax.set_title(f"{title}\n{ts_label}", fontsize=9)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax, fraction=0.04)
            col.pyplot(fig, use_container_width=True)
            plt.close(fig)

st.caption(f"SIH26077 · Hybrid TransUNet · Window {window_idx}/{n_windows-1} · Lead: {lead_time}h · {ts_label}")
