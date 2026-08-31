import { create } from 'zustand';
import { earlyWarningApi } from '../services/apiService';

// Fallback basic regions (just for the dropdown)
const defaultRegions = [
  { id: 'uk-hp', name: 'Uttarakhand & Himachal', center: [31.20, 78.40], zoom: 7 },
  { id: 'uk',    name: 'Uttarakhand',            center: [30.30, 79.20], zoom: 8 },
  { id: 'hp',    name: 'Himachal Pradesh',       center: [31.70, 77.20], zoom: 8 },
];

export const useAppStore = create((set, get) => ({

  // ── Global Dashboard Data (Async) ─────────────────────────────────
  isDataLoading: true,
  riskSummary: null,
  riskTimeline: [],
  xaiTriggers: [],
  alerts: [],
  dataSources: [],
  systemPerformance: [],
  districtRiskData: [],
  geoHeatmap: null,
  geoTrajectories: null,

  fetchDashboardData: async () => {
    set({ isDataLoading: true });
    try {
      const [
        summary,
        timeline,
        xai,
        alerts,
        sources,
        performance,
        districts,
        heatmap,
        trajectories
      ] = await Promise.all([
        earlyWarningApi.getRiskSummary(),
        earlyWarningApi.getRiskTimeline(),
        earlyWarningApi.getXaiTriggers(),
        earlyWarningApi.getAlerts(),
        earlyWarningApi.getDataSources(),
        earlyWarningApi.getSystemPerformance(),
        earlyWarningApi.getDistrictRiskData(),
        earlyWarningApi.getGeoHeatmap(),
        earlyWarningApi.getGeoTrajectories()
      ]);

      set({
        riskSummary: summary,
        riskTimeline: timeline,
        xaiTriggers: xai,
        alerts: alerts,
        dataSources: sources,
        systemPerformance: performance,
        districtRiskData: districts,
        geoHeatmap: heatmap,
        geoTrajectories: trajectories,
        isDataLoading: false
      });
    } catch (error) {
      console.error("Failed to fetch dashboard data:", error);
      set({ isDataLoading: false });
    }
  },

  // ── Region Selector ──────────────────────────────────────────────
  selectedRegion: 'uk-hp',
  regions: defaultRegions,
  setSelectedRegion: (regionId) => set({ selectedRegion: regionId }),

  // ── Time Range ───────────────────────────────────────────────────
  selectedTimeRange: '2h',
  setSelectedTimeRange: (range) => set({ selectedTimeRange: range }),

  // ── Map Data Layers ───────────────────────────────────
  mapLayers: {
    'imdaa-risk': true,
    'insat-cloud': false,
    'qpe-rain': false,
    'dem-terrain': false,
    'soil-moisture': false,
    'river-gauges': false,
    'past-incidents': false
  },
  toggleMapLayer: (layerKey) => set((state) => ({
    mapLayers: {
      ...state.mapLayers,
      [layerKey]: !state.mapLayers[layerKey],
    },
  })),
  setActiveLayer: (layerKey) => set((state) => {
    const next = {};
    Object.keys(state.mapLayers).forEach(k => {
      next[k] = k === layerKey;
    });
    return { mapLayers: next };
  }),

  basemap: 'satellite',
  setBasemap: (bm) => set({ basemap: bm }),
  showBorders: true,
  toggleBorders: () => set((s) => ({ showBorders: !s.showBorders })),

  // ── District Inspection ──────────────────────────────────────────
  selectedDistrict: null,
  setSelectedDistrict: (district) => set({ selectedDistrict: district }),

  // ── Alert Detail Modal ───────────────────────────────────────────
  selectedAlert: null,
  setSelectedAlert: (alert) => set({ selectedAlert: alert }),
  clearSelectedAlert: () => set({ selectedAlert: null }),

  acknowledgedAlerts: [],
  acknowledgeAlert: (alertId) => set((state) => ({
    acknowledgedAlerts: [...state.acknowledgedAlerts, alertId],
  })),

  // ── Search ───────────────────────────────────────────────────────
  searchQuery: '',
  setSearchQuery: (query) => set({ searchQuery: query }),

  // ── Reset ────────────────────────────────────────────────────────
  resetFilters: () => set({
    selectedRegion: 'uk-hp',
    selectedTimeRange: '2h',
    searchQuery: '',
    selectedDistrict: null,
    selectedAlert: null,
  }),
}));

