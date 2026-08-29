/**
 * Realistically shaped GeoJSON polygons for Uttarakhand & Himachal Pradesh districts
 * and multi-ring convective risk contours for meteorological nowcasting.
 */

export const UTTARAKHAND_DISTRICTS_GEOJSON = {
  type: "FeatureCollection",
  features: [
    // 1. Chamoli (Extreme / Very High Risk Core)
    {
      type: "Feature",
      properties: { name: "Chamoli", risk: "extreme", riskLevel: "Extreme (88%)", color: "#DC2626", stroke: "#EF4444" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [79.15, 30.75], [79.40, 31.00], [79.75, 30.95], [80.05, 30.70],
          [80.00, 30.35], [79.60, 30.25], [79.25, 30.30], [79.15, 30.55], [79.15, 30.75]
        ]]
      }
    },
    // 2. Rudraprayag (Very High Risk Zone)
    {
      type: "Feature",
      properties: { name: "Rudraprayag", risk: "veryhigh", riskLevel: "Very High (85%)", color: "#EA580C", stroke: "#F97316" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [78.85, 30.70], [79.20, 30.65], [79.25, 30.30], [78.88, 30.20],
          [78.75, 30.45], [78.85, 30.70]
        ]]
      }
    },
    // 3. Tehri Garhwal (High / Moderate Risk Zone)
    {
      type: "Feature",
      properties: { name: "Tehri Garhwal", risk: "high", riskLevel: "High (72%)", color: "#EAB308", stroke: "#FACC15" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [78.30, 30.65], [78.85, 30.70], [78.75, 30.20], [78.35, 30.05],
          [78.15, 30.35], [78.30, 30.65]
        ]]
      }
    },
    // 4. Uttarkashi (Moderate / High Risk Zone)
    {
      type: "Feature",
      properties: { name: "Uttarkashi", risk: "moderate", riskLevel: "Moderate (74%)", color: "#84CC16", stroke: "#A3E635" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [77.95, 31.40], [78.50, 31.45], [79.25, 31.25], [79.15, 30.75],
          [78.30, 30.65], [77.95, 30.90], [77.95, 31.40]
        ]]
      }
    },
    // 5. Bageshwar (High Risk Zone)
    {
      type: "Feature",
      properties: { name: "Bageshwar", risk: "high", riskLevel: "High (70%)", color: "#EA580C", stroke: "#F97316" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [79.60, 30.25], [80.05, 30.20], [80.00, 29.80], [79.60, 29.75],
          [79.55, 30.00], [79.60, 30.25]
        ]]
      }
    },
    // 6. Pithoragarh (Moderate / High Risk)
    {
      type: "Feature",
      properties: { name: "Pithoragarh", risk: "moderate", riskLevel: "Moderate (54%)", color: "#EAB308", stroke: "#FACC15" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [80.05, 30.60], [80.55, 30.35], [80.50, 29.60], [80.00, 29.55],
          [80.00, 30.20], [80.05, 30.60]
        ]]
      }
    },
    // 7. Almora (Moderate Risk)
    {
      type: "Feature",
      properties: { name: "Almora", risk: "moderate", riskLevel: "Moderate (48%)", color: "#84CC16", stroke: "#A3E635" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [79.25, 29.85], [79.80, 29.80], [79.75, 29.45], [79.20, 29.45],
          [79.25, 29.85]
        ]]
      }
    },
    // 8. Nainital (Low / Moderate Risk)
    {
      type: "Feature",
      properties: { name: "Nainital", risk: "low", riskLevel: "Low-Mod (42%)", color: "#22C55E", stroke: "#4ADE80" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [79.05, 29.55], [79.75, 29.45], [79.70, 29.10], [79.00, 29.15],
          [79.05, 29.55]
        ]]
      }
    },
    // 9. Champawat (Low Risk)
    {
      type: "Feature",
      properties: { name: "Champawat", risk: "low", riskLevel: "Low (28%)", color: "#16A34A", stroke: "#22C55E" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [79.80, 29.50], [80.30, 29.45], [80.25, 29.05], [79.75, 29.10],
          [79.80, 29.50]
        ]]
      }
    },
    // 10. Dehradun (Low Risk)
    {
      type: "Feature",
      properties: { name: "Dehradun", risk: "low", riskLevel: "Low (35%)", color: "#16A34A", stroke: "#22C55E" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [77.60, 30.65], [78.20, 30.65], [78.15, 30.00], [77.65, 30.05],
          [77.60, 30.65]
        ]]
      }
    },
    // 11. Haridwar (Low Risk)
    {
      type: "Feature",
      properties: { name: "Haridwar", risk: "low", riskLevel: "Low (20%)", color: "#15803D", stroke: "#16A34A" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [77.75, 30.05], [78.35, 30.00], [78.25, 29.60], [77.70, 29.65],
          [77.75, 30.05]
        ]]
      }
    },
    // 12. Pauri Garhwal (Moderate Risk)
    {
      type: "Feature",
      properties: { name: "Pauri Garhwal", risk: "moderate", riskLevel: "Moderate (45%)", color: "#EAB308", stroke: "#FACC15" },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [78.40, 30.20], [79.20, 30.25], [79.10, 29.65], [78.45, 29.70],
          [78.40, 30.20]
        ]]
      }
    }
  ]
};

// District pinned map label locations
export const DISTRICT_MAP_LABELS = [
  { name: 'Chamoli', lat: 30.55, lng: 79.48, isHotspot: true, riskText: 'Very High (88%)' },
  { name: 'Rudraprayag', lat: 30.38, lng: 78.98, isHotspot: true, riskText: 'Very High (85%)' },
  { name: 'Uttarkashi', lat: 30.85, lng: 78.50, isHotspot: false, riskText: 'Moderate (74%)' },
  { name: 'Tehri Garhwal', lat: 30.38, lng: 78.48, isHotspot: false, riskText: 'High (72%)' },
  { name: 'Bageshwar', lat: 29.92, lng: 79.78, isHotspot: false, riskText: 'High (70%)' },
  { name: 'Pithoragarh', lat: 29.85, lng: 80.20, isHotspot: false, riskText: 'Moderate (54%)' },
  { name: 'Almora', lat: 29.62, lng: 79.58, isHotspot: false, riskText: 'Moderate (48%)' },
  { name: 'Nainital', lat: 29.35, lng: 79.40, isHotspot: false, riskText: 'Low (42%)' },
  { name: 'Champawat', lat: 29.30, lng: 80.05, isHotspot: false, riskText: 'Low (28%)' },
  { name: 'Dehradun', lat: 30.30, lng: 77.95, isHotspot: false, riskText: 'Low (35%)' },
  { name: 'Haridwar', lat: 29.85, lng: 78.05, isHotspot: false, riskText: 'Low (20%)' },
  { name: 'Shimla', lat: 31.05, lng: 77.10, isHotspot: false, riskText: 'Low (32%)' },
  { name: 'Kullu', lat: 31.95, lng: 77.05, isHotspot: false, riskText: 'Moderate (55%)' },
  { name: 'Himachal Pradesh', lat: 31.85, lng: 77.85, isRegionText: true },
];

// Convective isohyet risk rings centered on the Rudraprayag / Chamoli cloudburst zone
export const CONVECTIVE_RISK_CIRCLES = [
  {
    name: "Extreme Cloudburst Epicenter",
    center: [30.48, 79.30],
    radius: 20000,
    color: "#7E22CE",
    fillColor: "#7E22CE",
    fillOpacity: 0.65,
    weight: 2,
  },
  {
    name: "Very High Cloudburst & Flash Flood Core",
    center: [30.45, 79.25],
    radius: 40000,
    color: "#DC2626",
    fillColor: "#DC2626",
    fillOpacity: 0.55,
    weight: 2,
  },
  {
    name: "High Severe Thunderstorm Zone",
    center: [30.40, 79.20],
    radius: 65000,
    color: "#EA580C",
    fillColor: "#EA580C",
    fillOpacity: 0.40,
    weight: 1.5,
  },
  {
    name: "Moderate Instability Fringe",
    center: [30.35, 79.15],
    radius: 95000,
    color: "#EAB308",
    fillColor: "#EAB308",
    fillOpacity: 0.28,
    weight: 1.5,
  },
  {
    name: "Low Pre-Convective Perimeter",
    center: [30.30, 79.10],
    radius: 135000,
    color: "#16A34A",
    fillColor: "#16A34A",
    fillOpacity: 0.18,
    weight: 1,
  },
];
