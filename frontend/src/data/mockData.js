/**
 * Mock data for SIH26077 - IMD AI-Driven Hyper-Local Early Warning System
 * All timestamps use 24-Hour Indian Standard Time (IST) format.
 */

// ═══════════════════════════════════════════════════════════════════
// REGIONS
// ═══════════════════════════════════════════════════════════════════

export const mockRegions = [
  { id: 'uk-hp', name: 'Uttarakhand & Himachal Pradesh', center: [31.20, 78.40], zoom: 7 },
  { id: 'uk',    name: 'Uttarakhand',                    center: [30.30, 79.20], zoom: 8 },
  { id: 'hp',    name: 'Himachal Pradesh',                center: [31.70, 77.20], zoom: 8 },
];

// ═══════════════════════════════════════════════════════════════════
// RISK SUMMARY — varies by time range
// ═══════════════════════════════════════════════════════════════════

export const mockRiskSummaryByTime = {
  now: {
    overallRisk:      { level: 'High',      affectedDistricts: 8,  gaugeValue: 68 },
    thunderstormRisk: { level: 'Moderate',  probability: 58 },
    cloudburstRisk:   { level: 'High',      probability: 72 },
    flashFloodRisk:   { level: 'Moderate',  probability: 55 },
  },
  '2h': {
    overallRisk:      { level: 'High',      affectedDistricts: 12, gaugeValue: 78 },
    thunderstormRisk: { level: 'High',      probability: 75 },
    cloudburstRisk:   { level: 'Very High', probability: 85 },
    flashFloodRisk:   { level: 'High',      probability: 70 },
  },
  '4h': {
    overallRisk:      { level: 'Very High', affectedDistricts: 14, gaugeValue: 86 },
    thunderstormRisk: { level: 'Very High', probability: 82 },
    cloudburstRisk:   { level: 'Very High', probability: 90 },
    flashFloodRisk:   { level: 'High',      probability: 78 },
  },
  '6h': {
    overallRisk:      { level: 'Moderate',  affectedDistricts: 6,  gaugeValue: 52 },
    thunderstormRisk: { level: 'Moderate',  probability: 48 },
    cloudburstRisk:   { level: 'Moderate',  probability: 55 },
    flashFloodRisk:   { level: 'Moderate',  probability: 45 },
  },
};

// Backward-compatible default
export const mockRiskSummary = mockRiskSummaryByTime['2h'];

// ═══════════════════════════════════════════════════════════════════
// RISK TIMELINE — 24-Hour format (10:00 to 16:00)
// ═══════════════════════════════════════════════════════════════════

export const mockRiskTimelineByTime = {
  now: [
    { time: '10:00', thunderstorm: 30, cloudburst: 50, flashFlood: 10 },
    { time: '10:30', thunderstorm: 28, cloudburst: 49, flashFlood: 12 },
    { time: '11:00', thunderstorm: 32, cloudburst: 52, flashFlood: 9 },
    { time: '11:30', thunderstorm: 40, cloudburst: 65, flashFlood: 20 },
    { time: '12:00', thunderstorm: 38, cloudburst: 63, flashFlood: 22 },
    { time: '12:30', thunderstorm: 45, cloudburst: 70, flashFlood: 28 },
    { time: '13:00', thunderstorm: 55, cloudburst: 80, flashFlood: 35 },
    { time: '13:30', thunderstorm: 50, cloudburst: 76, flashFlood: 32 },
    { time: '14:00', thunderstorm: 60, cloudburst: 83, flashFlood: 40 },
    { time: '14:30', thunderstorm: 56, cloudburst: 79, flashFlood: 38 },
    { time: '15:00', thunderstorm: 52, cloudburst: 81, flashFlood: 33 },
    { time: '15:30', thunderstorm: 58, cloudburst: 77, flashFlood: 36 },
    { time: '16:00', thunderstorm: 54, cloudburst: 80, flashFlood: 34 },
  ],
  '2h': [
    { time: '10:00', thunderstorm: 38, cloudburst: 58, flashFlood: 15 },
    { time: '10:30', thunderstorm: 37, cloudburst: 57, flashFlood: 16 },
    { time: '11:00', thunderstorm: 40, cloudburst: 60, flashFlood: 14 },
    { time: '11:30', thunderstorm: 51, cloudburst: 78, flashFlood: 27 },
    { time: '12:00', thunderstorm: 49, cloudburst: 76, flashFlood: 30 },
    { time: '12:30', thunderstorm: 58, cloudburst: 82, flashFlood: 38 },
    { time: '13:00', thunderstorm: 68, cloudburst: 91, flashFlood: 44 },
    { time: '13:30', thunderstorm: 64, cloudburst: 88, flashFlood: 42 },
    { time: '14:00', thunderstorm: 73, cloudburst: 94, flashFlood: 51 },
    { time: '14:30', thunderstorm: 70, cloudburst: 89, flashFlood: 49 },
    { time: '15:00', thunderstorm: 67, cloudburst: 92, flashFlood: 45 },
    { time: '15:30', thunderstorm: 72, cloudburst: 87, flashFlood: 48 },
    { time: '16:00', thunderstorm: 68, cloudburst: 90, flashFlood: 46 },
  ],
  '4h': [
    { time: '10:00', thunderstorm: 42, cloudburst: 64, flashFlood: 18 },
    { time: '10:30', thunderstorm: 40, cloudburst: 62, flashFlood: 20 },
    { time: '11:00', thunderstorm: 45, cloudburst: 66, flashFlood: 18 },
    { time: '11:30', thunderstorm: 56, cloudburst: 83, flashFlood: 32 },
    { time: '12:00', thunderstorm: 53, cloudburst: 80, flashFlood: 35 },
    { time: '12:30', thunderstorm: 62, cloudburst: 87, flashFlood: 43 },
    { time: '13:00', thunderstorm: 72, cloudburst: 96, flashFlood: 50 },
    { time: '13:30', thunderstorm: 68, cloudburst: 92, flashFlood: 47 },
    { time: '14:00', thunderstorm: 78, cloudburst: 98, flashFlood: 56 },
    { time: '14:30', thunderstorm: 75, cloudburst: 94, flashFlood: 54 },
    { time: '15:00', thunderstorm: 71, cloudburst: 97, flashFlood: 50 },
    { time: '15:30', thunderstorm: 76, cloudburst: 92, flashFlood: 53 },
    { time: '16:00', thunderstorm: 72, cloudburst: 95, flashFlood: 51 },
  ],
  '6h': [
    { time: '10:00', thunderstorm: 38, cloudburst: 58, flashFlood: 15 },
    { time: '10:30', thunderstorm: 37, cloudburst: 57, flashFlood: 16 },
    { time: '11:00', thunderstorm: 40, cloudburst: 60, flashFlood: 14 },
    { time: '11:30', thunderstorm: 51, cloudburst: 78, flashFlood: 27 },
    { time: '12:00', thunderstorm: 49, cloudburst: 76, flashFlood: 30 },
    { time: '12:30', thunderstorm: 58, cloudburst: 82, flashFlood: 38 },
    { time: '13:00', thunderstorm: 68, cloudburst: 91, flashFlood: 44 },
    { time: '13:30', thunderstorm: 64, cloudburst: 88, flashFlood: 42 },
    { time: '14:00', thunderstorm: 73, cloudburst: 94, flashFlood: 51 },
    { time: '14:30', thunderstorm: 70, cloudburst: 89, flashFlood: 49 },
    { time: '15:00', thunderstorm: 67, cloudburst: 92, flashFlood: 45 },
    { time: '15:30', thunderstorm: 72, cloudburst: 87, flashFlood: 48 },
    { time: '16:00', thunderstorm: 68, cloudburst: 90, flashFlood: 46 },
  ],
};

export const mockRiskTimeline = mockRiskTimelineByTime['6h'];

// ═══════════════════════════════════════════════════════════════════
// XAI TRIGGERS
// ═══════════════════════════════════════════════════════════════════

export const mockXaiTriggers = [
  {
    id: 'iwv',
    title: 'High Integrated Water Vapor (IWV)',
    description: 'Rapid moisture accumulation detected',
    impact: '+85%',
    impactType: 'danger',
    iconType: 'droplet',
  },
  {
    id: 'ctt',
    title: 'Cloud Top Temperature Drop',
    description: 'Strong updrafts indicating storm development',
    impact: '-6.2°C/hr',
    impactType: 'danger',
    iconType: 'cloud-snow',
  },
  {
    id: 'capecin',
    title: 'High CAPE / Low CIN',
    description: 'Atmosphere highly unstable',
    impact: '+72%',
    impactType: 'danger',
    iconType: 'zap',
  },
  {
    id: 'convergence',
    title: 'Low Level Convergence',
    description: 'Wind convergence detected at surface',
    impact: 'Strong',
    impactType: 'danger',
    iconType: 'wind',
  },
  {
    id: 'shear',
    title: 'Vertical Wind Shear',
    description: 'Favorable storm tilt & longevity',
    impact: 'Elevated',
    impactType: 'danger',
    iconType: 'compass',
  },
  {
    id: 'terrain',
    title: 'Terrain Influence',
    description: 'Steep slope & drainage basin aligned',
    impact: 'High',
    impactType: 'danger',
    iconType: 'mountain',
  },
];

// ═══════════════════════════════════════════════════════════════════
// RECENT ALERTS — 24-Hour IST format
// ═══════════════════════════════════════════════════════════════════

export const mockRecentAlerts = [
  {
    id: 'alert-1',
    type: 'Cloudburst Warning',
    location: 'Chamoli, Uttarakhand',
    state: 'Uttarakhand',
    time: '10:20 IST',
    severity: 'danger',
    details: 'Heavy localized precipitation exceeding 85mm/hr forecast within the next 2 hours. Catchment basins on high alert.',
    issuedBy: 'IMD Regional Office – Dehradun',
    validUntil: '12:30 IST',
    expectedRainfall: '80–120 mm',
    windSpeed: '40–60 km/h',
    affectedPopulation: '~3,91,605',
    recommendedAction: 'Evacuate low-lying areas near Alaknanda river. Close mountain roads. Alert SDRF teams.',
    rivers: 'Alaknanda, Dhauliganga',
  },
  {
    id: 'alert-2',
    type: 'Flash Flood Warning',
    location: 'Rudraprayag, Uttarakhand',
    state: 'Uttarakhand',
    time: '10:18 IST',
    severity: 'danger',
    details: 'Alaknanda and Mandakini river catchments approaching danger mark. Low-lying settlements advised immediate evacuation.',
    issuedBy: 'IMD Central Warning Office',
    validUntil: '14:00 IST',
    expectedRainfall: '60–100 mm',
    windSpeed: '30–50 km/h',
    affectedPopulation: '~2,42,285',
    recommendedAction: 'Suspend downstream bridge operations. Deploy rescue boats. Establish emergency shelters on higher ground.',
    rivers: 'Mandakini, Alaknanda',
  },
  {
    id: 'alert-3',
    type: 'Thunderstorm Alert',
    location: 'Mandi, Himachal Pradesh',
    state: 'Himachal Pradesh',
    time: '10:15 IST',
    severity: 'warning',
    details: 'Severe convective cloud clusters with frequent lightning strikes and gusty surface winds (60-70 km/h).',
    issuedBy: 'IMD Regional Office – Shimla',
    validUntil: '13:00 IST',
    expectedRainfall: '40–60 mm',
    windSpeed: '60–70 km/h',
    affectedPopulation: '~9,99,777',
    recommendedAction: 'Stay indoors. Avoid open fields and tall structures. Unplug electronic devices.',
    rivers: 'Beas, Suketi',
  },
  {
    id: 'alert-4',
    type: 'Intense Rain Watch',
    location: 'Uttarkashi, Uttarakhand',
    state: 'Uttarakhand',
    time: '09:50 IST',
    severity: 'warning',
    details: 'Elevated moisture flux convergence along the Bhagirathi river basin. Sustained heavy rainfall expected for 3–4 hours.',
    issuedBy: 'IMD Nowcast Division',
    validUntil: '13:30 IST',
    expectedRainfall: '50–80 mm',
    windSpeed: '30–45 km/h',
    affectedPopulation: '~3,30,086',
    recommendedAction: 'Monitor river gauge levels. Pre-position rescue equipment along Bhagirathi settlements.',
    rivers: 'Bhagirathi, Yamuna',
  },
  {
    id: 'alert-5',
    type: 'Slope Inundation Watch',
    location: 'Kullu, Himachal Pradesh',
    state: 'Himachal Pradesh',
    time: '09:30 IST',
    severity: 'info',
    details: 'Saturated soil profile increases landslide and debris flow vulnerability under persistent downpours.',
    issuedBy: 'GSI/IMD Joint Advisory',
    validUntil: '16:00 IST',
    expectedRainfall: '30–50 mm',
    windSpeed: '20–35 km/h',
    affectedPopulation: '~4,37,903',
    recommendedAction: 'Avoid travel on hillside roads. Watch for cracks in slope surfaces. Report unusual seepage to SDMA.',
    rivers: 'Beas, Parbati, Tirthan',
  },
];

// ═══════════════════════════════════════════════════════════════════
// DATA SOURCES
// ═══════════════════════════════════════════════════════════════════

export const mockDataSources = [
  {
    id: 'insat',
    name: 'INSAT-3D/3DR Satellite',
    status: 'Live',
    type: 'live',
    frequency: 'Every 15 min',
    lastSync: '2 min ago',
  },
  {
    id: 'imdaa',
    name: 'IMDAA Reanalysis',
    status: 'Live',
    type: 'live',
    frequency: 'Hourly',
    lastSync: '12 min ago',
  },
  {
    id: 'qpe',
    name: 'QPE (Satellite)',
    status: 'Live',
    type: 'live',
    frequency: 'Every 15 min',
    lastSync: '4 min ago',
  },
  {
    id: 'dem',
    name: 'Digital Elevation Model',
    status: 'Static',
    type: 'static',
    frequency: '30m SRTM',
    lastSync: 'Baseline v3',
  },
];

// ═══════════════════════════════════════════════════════════════════
// SYSTEM PERFORMANCE
// ═══════════════════════════════════════════════════════════════════

export const mockSystemPerformance = [
  { name: 'Model Confidence',      percentage: 92, status: 'optimal' },
  { name: 'Data Availability',     percentage: 98, status: 'optimal' },
  { name: 'Prediction Accuracy (7d)', percentage: 89, status: 'optimal' },
];

// ═══════════════════════════════════════════════════════════════════
// MAP LAYER DEFINITIONS
// ═══════════════════════════════════════════════════════════════════

export const MAP_LAYER_DEFINITIONS = [
  { key: 'risk',       label: 'Risk Level',              icon: 'shield',      defaultOn: true },
  { key: 'precip',     label: 'Precipitation',           icon: 'cloud-rain',  defaultOn: false },
  { key: 'iwv',        label: 'Integrated Water Vapor',  icon: 'droplet',     defaultOn: false },
  { key: 'ctt',        label: 'Cloud Top Temperature',   icon: 'thermometer', defaultOn: false },
  { key: 'cape',       label: 'CAPE',                    icon: 'zap',         defaultOn: false },
  { key: 'wind',       label: 'Wind',                    icon: 'wind',        defaultOn: false },
  { key: 'dem',        label: 'DEM (Terrain)',            icon: 'mountain',    defaultOn: false },
];
