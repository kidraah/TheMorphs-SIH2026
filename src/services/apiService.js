import axios from 'axios';
import {
  mockRiskSummary,
  mockRiskTimeline,
  mockXaiTriggers,
  mockRecentAlerts,
  mockDataSources,
  mockSystemPerformance,
} from '../data/mockData';
import { DISTRICT_RISK_DATA } from '../data/districtData';

// Base Axios instance configured for future IMD / MoES API integration
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const earlyWarningApi = {
  // Get overall risk metrics
  getRiskSummary: async () => {
    // In future: return (await apiClient.get('/nowcast/summary')).data;
    return Promise.resolve(mockRiskSummary);
  },

  // Get 6-hour risk timeline
  getRiskTimeline: async (region = 'uk-hp') => {
    return Promise.resolve(mockRiskTimeline);
  },

  // Get Explainable AI feature weights
  getXaiTriggers: async () => {
    return Promise.resolve(mockXaiTriggers);
  },

  // Get active alerts
  getAlerts: async () => {
    return Promise.resolve(mockRecentAlerts);
  },

  // Get ingested telemetry sources
  getDataSources: async () => {
    return Promise.resolve(mockDataSources);
  },

  // Get model performance telemetry
  getSystemPerformance: async () => {
    return Promise.resolve(mockSystemPerformance);
  },

  // Get all district details
  getDistrictRiskData: async () => {
    return Promise.resolve(DISTRICT_RISK_DATA);
  },
};

export default apiClient;
