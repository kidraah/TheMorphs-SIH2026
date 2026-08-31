# VARUNA AI: Hyper-Local Early Warning System 🌩️🚨

**VARUNA AI** is an AI-driven, hyper-local early warning system designed for the **nowcasting (2-6 hours)** of severe thunderstorms, cloudbursts, and flash floods. Built for the India Meteorological Department (IMD) and disaster management authorities, VARUNA AI fuses multi-modal meteorological data through a Hybrid Spatiotemporal TransUNet architecture to deliver highly accurate, actionable, and explainable risk assessments.

---

## 🏗️ System Architecture

The architecture is divided into four main pillars: Data Sources, Pipeline, the VARUNA Model, and Delivery.

```mermaid
graph TD
    %% Define Styles
    classDef source fill:#e8f5e9,stroke:#81c784,stroke-width:2px;
    classDef pipeline fill:#e3f2fd,stroke:#64b5f6,stroke-width:2px;
    classDef model fill:#f3e5f5,stroke:#ba68c8,stroke-width:2px;
    classDef delivery fill:#fff3e0,stroke:#ffb74d,stroke-width:2px;

    subgraph Sources ["📡 1. Multi-Modal Data Sources"]
        A1["DYNAMIC - REANALYSIS<br>IMDAA: Temp, Humidity (CAPE/CIN), U/V Winds"]:::source
        A2["SATELLITE - INSAT-3D/3DR<br>WV (IWV), TIR (CTT drop), QPE (MOSDAC)"]:::source
        A3["STATIC - TERRAIN<br>CartoDEM/SRTM: Elevation, Slope, Basins"]:::source
    end

    subgraph Pipeline ["⚙️ 2. Data Fusion & Pipeline"]
        B1["Fusion & Alignment<br>Normalize & map to unified spatiotemporal grid"]:::pipeline
        B2[("Storage<br>Aligned grids (Training/Validation)")]:::pipeline
        B3["Predictive Matrix<br>Moisture, Instability, Lift, Terrain"]:::pipeline
        B4["BACKEND<br>APIs & Scheduled Jobs"]:::pipeline
        
        B1 --> B2
        B2 --> B3
    end

    subgraph Varuna ["🧠 3. VARUNA (MODEL)"]
        C1["Shared Backbone<br>Spatiotemporal Transformer (Cross-Attention)"]:::model
        C2["Multi-Task Heads (2-6 hr)<br>Thunderstorm, Cloudburst, Flash Flood"]:::model
        C3["XAI Layer<br>Meteorological triggers & explainability"]:::model
        C4["Threshold Engine<br>Breach detection & categorized alerts"]:::model
        
        C1 --> C2
        C2 --> C3
        C3 --> C4
    end

    subgraph Delivery ["📱 4. Delivery to Responders"]
        D1["SPATIAL DASHBOARD<br>Web console for authorities"]:::delivery
        D2["RISK MAPS<br>Hyper-local probability layers on DEM"]:::delivery
        D3["AUTOMATED ALERTS<br>Warnings to first responders"]:::delivery
    end

    %% Connections
    A1 --> B1
    A2 --> B1
    A3 --> B1
    
    B3 -->|Feature Grids| C1
    
    B4 --> D1
    B4 --> D2
    
    C4 --> D3
```

---

## 🌟 Key Features

- **Multi-Modal Data Fusion**: Assimilates 30+ atmospheric variables from IMDAA, 4-channel INSAT-3D satellite imagery, and high-resolution SRTM DEM terrain data.
- **Hybrid TransUNet Architecture**: Utilizes 3D temporal compression, a shared CNN-Transformer bottleneck with Multi-Head Self-Attention for global receptive awareness, and independent multi-task U-Net decoders.
- **Explainable AI (XAI)**: Demystifies model predictions by breaking down alerts into readable meteorological triggers (e.g., CAPE threshold breaches, rapid CTT drops).
- **Official SOP Reporting**: Automatically generates standard operating procedure (SOP) compliant PDF bulletins for NDMA, SDMA, and District Collectors based on live tensor outputs.
- **Geospatial Risk Mapping**: Real-time rendering of extreme risk probabilities overlaid directly onto MapLibre GL JS mapping engines with district-level zonal boundary detection.

---

## 🛠️ Technology Stack

### Frontend (User Interface)
- **Framework**: React.js with Vite
- **Styling**: Tailwind CSS, Lucide Icons
- **State Management**: Zustand
- **Mapping Engine**: MapLibre GL JS, Deck.gl (BitmapLayer, GeoJsonLayer)
- **Routing**: React Router DOM

### Backend (API & Orchestration)
- **Framework**: FastAPI (Python)
- **Geospatial Processing**: Rasterio, Rasterstats, SciPy (Gaussian smoothing)
- **Concurrency**: Threading locks for safe parallel inference

### Deep Learning Model
- **Framework**: PyTorch
- **Architecture**: Hybrid TransUNet (3D Convolutions + Vision Transformer + U-Net Decoders)
- **Pre-trained Artifacts**: `sample_inference_data.pt`

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.9+)

### 1. Backend Setup
Navigate to the `backend` directory, set up your virtual environment, and start the FastAPI server:
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate  # On Windows
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
The backend API will run on `http://localhost:8000`.

### 2. Frontend Setup
Navigate to the `frontend` directory, install dependencies, and start the Vite dev server:
```bash
cd frontend
npm install
npm run dev
```
The application will be accessible at `http://localhost:3000`.

---

## 📡 Data Sources
VARUNA AI relies on the following active telemetry and static datasets:
1. **INSAT-3D/3DR Satellite**: Water Vapor (WV), Thermal Infrared (TIR), and QPE.
2. **IMDAA High-Res Reanalysis**: Temperature profiles, CAPE/CIN, U/V winds.
3. **Doppler Weather Radar (DWR)**: Real-time high-resolution precipitation tracking.
4. **Automated Rain Gauge (ARG)**: Ground truth validation and real-time river telemetry.
5. **SRTM Digital Elevation Model**: Static terrain, elevation, slope, and drainage basins.

---
*Developed for SIH 2026 - Problem Statement #77 (MoES/IMD)*
