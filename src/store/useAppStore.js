import { create } from 'zustand';
import { mockRecentAlerts, mockRegions, MAP_LAYER_DEFINITIONS } from '../data/mockData';
import { DISTRICT_RISK_DATA } from '../data/districtRiskData';

/**
 * Global application state for the IMD Early Warning Dashboard.
 *
 * Manages: region selection, time range, map layers, district inspection,
 * alerts, and search.
 */
export const useAppStore = create((set, get) => ({

  // ── Region Selector ──────────────────────────────────────────────
  selectedRegion: 'uk-hp',
  regions: mockRegions,
  setSelectedRegion: (regionId) => set({ selectedRegion: regionId }),

  // ── Time Range ───────────────────────────────────────────────────
  selectedTimeRange: '2h',   // 'now' | '2h' | '4h' | '6h'
  setSelectedTimeRange: (range) => set({ selectedTimeRange: range }),

  // ── Map Data Layers (7 layers) ───────────────────────────────────
  mapLayers: Object.fromEntries(
    MAP_LAYER_DEFINITIONS.map((l) => [l.key, l.defaultOn])
  ),
  toggleMapLayer: (layerKey) => set((state) => ({
    mapLayers: {
      ...state.mapLayers,
      [layerKey]: !state.mapLayers[layerKey],
    },
  })),
  setActiveLayer: (layerKey) => set((state) => {
    // Turn on the chosen layer, turn off all others (radio behavior)
    const next = {};
    MAP_LAYER_DEFINITIONS.forEach((l) => {
      next[l.key] = l.key === layerKey;
    });
    return { mapLayers: next };
  }),

  // Keep backward compat for basemap toggles
  basemap: 'satellite',   // 'satellite' | 'osm' | 'topo'
  setBasemap: (bm) => set({ basemap: bm }),

  showBorders: true,
  toggleBorders: () => set((s) => ({ showBorders: !s.showBorders })),

  // ── District Inspection ──────────────────────────────────────────
  selectedDistrict: null,
  setSelectedDistrict: (district) => set({ selectedDistrict: district }),

  // ── Alert Detail Modal ───────────────────────────────────────────
  alerts: mockRecentAlerts,
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
