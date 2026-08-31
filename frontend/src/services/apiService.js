import axios from 'axios';

// Base Axios instance configured for the real FastAPI backend
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const earlyWarningApi = {
  // Get overall risk metrics (driven by HF model)
  getRiskSummary: async () => {
    try { return (await apiClient.get('/api/risk/current')).data; } catch { return null; }
  },

  // Get 6-hour risk timeline (driven by HF model)
  getRiskTimeline: async (region = 'uk-hp') => {
    try { return (await apiClient.get('/api/risk/timeline')).data; } catch { return []; }
  },

  // Get Explainable AI feature weights (driven by HF model)
  getXaiTriggers: async () => {
    try { return (await apiClient.get('/api/xai/triggers')).data; } catch { return []; }
  },

  // Get active alerts (driven by HF model thresholds)
  getAlerts: async () => {
    try { return (await apiClient.get('/api/alerts')).data; } catch { return []; }
  },

  // Get ingested telemetry sources
  getDataSources: async () => {
    try { return (await apiClient.get('/api/data-sources/status')).data; } catch { return []; }
  },

  // Get model performance telemetry
  getSystemPerformance: async () => {
    try { return (await apiClient.get('/api/system/performance')).data; } catch { return []; }
  },

  // Get all district details
  getDistrictRiskData: async () => {
    try {
      const res = await apiClient.get('/api/risk/districts');
      return res.data;
    } catch {
      return [];
    }
  },

  getGeoHeatmap: async () => {
    try {
      return (await apiClient.get('/api/geo/heatmap')).data;
    } catch {
      return null;
    }
  },

  getGeoTrajectories: async () => {
    try {
      return (await apiClient.get('/api/geo/trajectories')).data;
    } catch {
      return null;
    }
  },
};

export default apiClient;
