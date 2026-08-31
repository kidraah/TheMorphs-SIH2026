# 🌩️ AI-Driven Hyper-Local Early Warning System

> **Smart India Hackathon 2026 — Problem Statement SIH77**  
> Ministry of Earth Sciences | India Meteorological Department (IMD)

A real-time, AI-powered early warning web platform for **Severe Thunderstorms**, **Cloudburst**, and **Flash Flood** nowcasting across India — built for disaster preparedness and hyper-local risk monitoring.

---

## 🚀 Live Demo

> Branch: [`riddhi`](https://github.com/kidraah/TheMorphs-SIH2026/tree/riddhi)  
> Team: **The Morphs**

---

## 📌 Problem Statement

India's disaster management agencies need **hyper-local, real-time alerts** for short-duration extreme weather events like cloudbursts, flash floods, and severe thunderstorms. This platform provides:

- ⚡ **Nowcasting** — 0–3 hour predictions at district/sub-district level
- 🗺️ **Interactive Risk Maps** — Live visualization of hazard zones using GeoJSON overlays
- 🔔 **Automated Alerts** — Color-coded severity alerts (Red / Orange / Yellow)
- 📊 **Explainable AI** — AI decision confidence scores for each prediction
- 📈 **Trend Analysis** — Historical risk timelines and precipitation data

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend Framework | React 18 + Vite |
| Routing | React Router v6 |
| State Management | Zustand |
| Maps | Leaflet + React-Leaflet |
| Charts | Recharts |
| Styling | Tailwind CSS |
| HTTP Client | Axios |
| Icons | Lucide React |

---

## 📁 Project Structure

```
SIH77_site/
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── RiskMap.jsx         # Interactive Leaflet risk map
│   │   ├── RiskCards.jsx       # District-wise risk summary cards
│   │   ├── AlertBanner.jsx     # Real-time alert banners
│   │   ├── AlertDetailModal.jsx
│   │   ├── PrecipitationNowcast.jsx
│   │   ├── ExplainableAI.jsx   # AI confidence + explainability panel
│   │   ├── RiskTimeline.jsx    # Historical risk chart
│   │   ├── Navbar.jsx / Sidebar.jsx
│   │   └── ...
│   ├── pages/              # Route-level page components
│   │   ├── Dashboard.jsx       # Main overview dashboard
│   │   ├── LiveMapPage.jsx     # Full-screen live risk map
│   │   ├── RiskOutlookPage.jsx # 72-hour risk outlook
│   │   ├── AlertsPage.jsx      # Alert management
│   │   ├── ForecastPage.jsx    # Extended forecast
│   │   ├── DataLayersPage.jsx  # Toggle data overlays
│   │   ├── ReportsPage.jsx     # Downloadable reports
│   │   ├── SettingsPage.jsx    # User preferences
│   │   └── AboutSystemPage.jsx # System methodology & info
│   ├── store/              # Zustand global state
│   ├── services/           # API service layer
│   ├── hooks/              # Custom React hooks
│   ├── utils/              # Formatters, helpers
│   ├── geojson/            # Himalayan district boundaries
│   └── data/               # Static/mock data
├── public/
├── index.html
├── vite.config.js
├── tailwind.config.js
└── package.json
```

---

## ⚙️ Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) v18+
- npm v9+

### Installation

```bash
# Clone the repository
git clone https://github.com/kidraah/TheMorphs-SIH2026.git
cd TheMorphs-SIH2026

# Switch to the development branch
git checkout riddhi

# Install dependencies
npm install

# Start the development server
npm run dev
```

The app will be available at `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

---

## 🗺️ Key Features

### 🏠 Dashboard
Central overview with live precipitation nowcast, active alert count, system performance metrics, and recent alert feed.

### 🗺️ Live Map
Interactive Leaflet map with toggleable data layers — risk zones, rain radar, district overlays — powered by Himalayan district GeoJSON boundaries.

### ⚠️ Risk Outlook
72-hour risk forecast with color-coded district-level risk cards and temporal trend charts.

### 🔔 Alerts
Filterable alert list with detail modals showing event type, severity level, affected regions, and recommended actions.

### 📊 Forecast
Extended weather parameters with precipitation probability, wind speed, and lightning risk scores.

### 🧠 Explainable AI
Visualizes the AI model's confidence score, contributing features, and uncertainty ranges for each prediction — ensuring transparency for disaster management officials.

### 📄 Reports
Generate and download district-level weather event reports.

---

## 👥 Team — The Morphs

| Role | Member |
|---|---|
| Team Lead / Full Stack | Riddhi |
| UI/UX & Frontend | — |
| ML / AI Model | — |
| Data Integration | — |
| Backend / APIs | — |

> *This project was built for Smart India Hackathon 2026.*

---

## 📜 License

This project is developed for academic/hackathon purposes under the guidance of the **Ministry of Earth Sciences, Government of India**.

---

## 🙏 Acknowledgements

- [India Meteorological Department (IMD)](https://mausam.imd.gov.in/)
- [National Disaster Management Authority (NDMA)](https://ndma.gov.in/)
- [OpenStreetMap Contributors](https://www.openstreetmap.org/)
- [Leaflet.js](https://leafletjs.com/)
