/**
 * Modular Map Layer & Basemap configurations for the GIS Risk Portal.
 * Separates Base Tile Layers from Data Overlays.
 */

export const BASEMAPS = [
  {
    id: 'satellite',
    name: 'Satellite View',
    type: 'base',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    maxZoom: 18,
    attribution: 'Esri World Imagery',
    description: 'High-resolution satellite imagery for physical terrain inspection',
  },
  {
    id: 'osm',
    name: 'Normal / Street Map',
    type: 'base',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    maxZoom: 18,
    attribution: 'OpenStreetMap',
    description: 'Standard cartographic view showing roads and settlements',
  },
  {
    id: 'topo',
    name: 'Terrain / DEM View',
    type: 'base',
    url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    maxZoom: 17,
    attribution: 'OpenTopoMap / SRTM DEM',
    description: 'Topographic contour lines and elevation shading for flood slope analysis',
  },
];

export const MAP_OVERLAYS = [
  {
    key: 'risk',
    name: 'Risk Level (Choropleth)',
    type: 'overlay',
    description: 'Convective nowcast risk classification polygons',
    defaultOn: true,
  },
  {
    key: 'precip',
    name: 'Precipitation Radar',
    type: 'overlay',
    description: 'QPE Doppler radar rainfall reflectivity',
    defaultOn: false,
  },
  {
    key: 'iwv',
    name: 'Integrated Water Vapor',
    type: 'overlay',
    description: 'INSAT-3D total column water vapor flux',
    defaultOn: false,
  },
  {
    key: 'ctt',
    name: 'Cloud Top Temperature',
    type: 'overlay',
    description: 'Infrared cloud top temperature soundings',
    defaultOn: false,
  },
];
