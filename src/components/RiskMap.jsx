import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Marker,
  useMap,
  useMapEvents,
} from 'react-leaflet';
import L from 'leaflet';
import {
  Plus, Minus, Crosshair, Layers as LayersIcon, Info,
  ShieldAlert, CloudRain, Droplet, Thermometer, Zap, Wind, Mountain,
} from 'lucide-react';
import { HIMALAYAN_DISTRICTS_GEOJSON } from '../geojson/himalayanDistricts';
import { DISTRICT_RISK_DATA } from '../data/districtRiskData';
import { PLACE_LABELS } from '../data/placeLabels';
import { MAP_LAYER_DEFINITIONS } from '../data/mockData';
import { BASEMAPS } from '../data/mapLayers';
import { MapLayerControl } from './MapLayerControl';
import { useAppStore } from '../store/useAppStore';
import { getCurrentISTTime } from '../utils/timeFormatter';

// ─── Constants ────────────────────────────────────────────────────

const RISK_STYLE = {
  LOW:         { fill: '#16A34A', opacity: 0.32, border: '#15803D' },
  MODERATE:    { fill: '#CA8A04', opacity: 0.42, border: '#A16207' },
  HIGH:        { fill: '#EA580C', opacity: 0.50, border: '#C2410C' },
  'VERY HIGH': { fill: '#DC2626', opacity: 0.56, border: '#B91C1C' },
  EXTREME:     { fill: '#7E22CE', opacity: 0.62, border: '#6B21A8' },
};

const ALERT_COLOR = {
  'Green Advisory': '#16A34A',
  'Yellow Watch':   '#CA8A04',
  'Orange Alert':   '#EA580C',
  'Red Warning':    '#DC2626',
};

const LEGEND_ITEMS = [
  { level: 'LOW',       color: '#16A34A', label: 'Low Risk' },
  { level: 'MODERATE',  color: '#CA8A04', label: 'Moderate Risk' },
  { level: 'HIGH',      color: '#EA580C', label: 'High Risk' },
  { level: 'VERY HIGH', color: '#DC2626', label: 'Very High Risk' },
  { level: 'EXTREME',   color: '#7E22CE', label: 'Extreme Risk' },
];

// Layer → overlay color
const OVERLAY_COLORS = {
  risk:   null,   // uses choropleth
  precip: '#0EA5E9',
  iwv:    '#6366F1',
  ctt:    '#8B5CF6',
  cape:   '#F59E0B',
  wind:   '#14B8A6',
  dem:    '#78716C',
};

// Pre-built lookup index
const RISK_INDEX = Object.fromEntries(
  DISTRICT_RISK_DATA.map((d) => [d.district.toLowerCase(), d])
);

// Helper for place label DivIcon
const createPlaceLabelIcon = (name, isHotspot = false) => {
  const fontWeight = isHotspot ? '800' : '600';
  const fontSize = isHotspot ? '11.5px' : '11px';

  return L.divIcon({
    className: 'custom-place-label-marker',
    html: `<div style="
      color: #FFFFFF;
      font-size: ${fontSize};
      font-weight: ${fontWeight};
      font-family: 'Inter', -apple-system, sans-serif;
      text-shadow: 0 1px 3px rgba(0,0,0,0.95), 0 0 5px rgba(0,0,0,0.85);
      white-space: nowrap;
      pointer-events: none;
      user-select: none;
      letter-spacing: 0.01em;
    ">${name}</div>`,
    iconSize: [110, 20],
    iconAnchor: [55, 10],
  });
};

// Layer icon component helper
const LayerIcon = ({ iconKey, className = 'w-3.5 h-3.5' }) => {
  switch (iconKey) {
    case 'shield':      return <ShieldAlert className={className} />;
    case 'cloud-rain':  return <CloudRain className={className} />;
    case 'droplet':     return <Droplet className={className} />;
    case 'thermometer': return <Thermometer className={className} />;
    case 'zap':         return <Zap className={className} />;
    case 'wind':        return <Wind className={className} />;
    case 'mountain':    return <Mountain className={className} />;
    default:            return <Layers className={className} />;
  }
};

// ─── Map view controller ──────────────────────────────────────────

const MapViewController = ({ center, zoom }) => {
  const map = useMap();
  React.useEffect(() => {
    map.setView(center, zoom, { animate: true });
  }, [center, zoom, map]);
  return null;
};

// ─── Zoom-dependent Place Markers Controller ──────────────────────

const PlaceMarkersController = ({ selectedRegion }) => {
  const map = useMap();
  const [currentZoom, setCurrentZoom] = useState(map.getZoom());

  useMapEvents({
    zoomend() {
      setCurrentZoom(map.getZoom());
    },
  });

  useEffect(() => {
    setCurrentZoom(map.getZoom());
  }, [map]);

  const visibleLabels = useMemo(() => {
    return PLACE_LABELS.filter((item) => {
      if (selectedRegion === 'uk' && item.state !== 'Uttarakhand') return false;
      if (selectedRegion === 'hp' && item.state !== 'Himachal Pradesh') return false;
      return currentZoom >= item.minZoom;
    });
  }, [selectedRegion, currentZoom]);

  return (
    <>
      {visibleLabels.map((place) => (
        <Marker
          key={place.name}
          position={[place.lat, place.lng]}
          icon={createPlaceLabelIcon(place.name, place.isHotspot)}
          interactive={false}
          zIndexOffset={1000}
        />
      ))}
    </>
  );
};

// ─── Main Component ───────────────────────────────────────────────

export const RiskMap = ({ height = 'h-[520px]', showControls = true }) => {
  const mapRef = useRef(null);
  const geoJsonRef = useRef(null);

  const {
    selectedRegion, regions,
    mapLayers, toggleMapLayer,
    basemap, setBasemap,
    showBorders, toggleBorders,
    setSelectedDistrict,
  } = useAppStore();

  const [showLayerPanel, setShowLayerPanel] = useState(false);
  const [hoveredDistrict, setHoveredDistrict] = useState(null);

  // Determine map center/zoom from selected region
  const region = regions.find((r) => r.id === selectedRegion) || regions[0];
  const mapCenter = region.center;
  const mapZoom = region.zoom;

  // Determine active overlay key (the first "on" layer besides risk)
  const activeOverlay = useMemo(() => {
    for (const key of Object.keys(mapLayers)) {
      if (key !== 'risk' && mapLayers[key]) return key;
    }
    return null;
  }, [mapLayers]);

  // Filter GeoJSON features by region
  const filteredGeoJSON = useMemo(() => {
    if (selectedRegion === 'uk-hp') return HIMALAYAN_DISTRICTS_GEOJSON;
    const stateFilter = selectedRegion === 'uk' ? 'Uttarakhand' : 'Himachal Pradesh';
    return {
      ...HIMALAYAN_DISTRICTS_GEOJSON,
      features: HIMALAYAN_DISTRICTS_GEOJSON.features.filter(
        (f) => f.properties.state === stateFilter
      ),
    };
  }, [selectedRegion]);

  // ── Choropleth style ────────────────────────────────────────────
  const getDistrictStyle = useCallback((feature) => {
    const key = (feature.properties.districtKey || '').toLowerCase();
    const data = RISK_INDEX[key];

    // Risk layer active → use risk colors
    if (mapLayers.risk) {
      const { fill, opacity, border } = (data && RISK_STYLE[data.riskLevel]) || RISK_STYLE.LOW;
      return {
        fillColor: fill,
        fillOpacity: opacity,
        color: showBorders ? border : 'transparent',
        weight: showBorders ? 1.4 : 0,
        dashArray: showBorders ? '4, 5' : null,
      };
    }

    // Non-risk overlay active → tint districts with overlay color
    if (activeOverlay && OVERLAY_COLORS[activeOverlay]) {
      const overlayColor = OVERLAY_COLORS[activeOverlay];
      // Use the data probabilities to vary opacity
      let intensity = 0.2;
      if (data) {
        const avg = (data.thunderstormProbability + data.cloudburstProbability + data.flashFloodProbability) / 3;
        intensity = 0.15 + (avg / 100) * 0.50;
      }
      return {
        fillColor: overlayColor,
        fillOpacity: intensity,
        color: showBorders ? overlayColor : 'transparent',
        weight: showBorders ? 1.2 : 0,
        dashArray: showBorders ? '4, 5' : null,
      };
    }

    // No overlay → just borders
    return {
      fillColor: 'transparent',
      fillOpacity: 0,
      color: showBorders ? '#94A3B8' : 'transparent',
      weight: showBorders ? 1.2 : 0,
      dashArray: showBorders ? '4, 5' : null,
    };
  }, [mapLayers, showBorders, activeOverlay]);

  // ── Hover highlight ─────────────────────────────────────────────
  const highlightFeature = useCallback((e) => {
    const layer = e.target;
    const key = (layer.feature.properties.districtKey || '').toLowerCase();
    const data = RISK_INDEX[key];
    const riskStyle = (data && RISK_STYLE[data.riskLevel]) || RISK_STYLE.LOW;

    layer.setStyle({
      weight: 2.5,
      color: riskStyle.border,
      fillOpacity: Math.min((riskStyle.opacity || 0.32) + 0.18, 0.78),
    });
    layer.bringToFront();
    setHoveredDistrict(data || null);
  }, []);

  const resetHighlight = useCallback((e) => {
    geoJsonRef.current?.resetStyle(e.target);
    setHoveredDistrict(null);
  }, []);

  // ── Click → popup + store update ────────────────────────────────
  const onFeatureClick = useCallback((e) => {
    const key = (e.target.feature.properties.districtKey || '').toLowerCase();
    const d = RISK_INDEX[key];
    if (!d) return;

    // Update global store with selected district
    setSelectedDistrict(d);

    const alertColor = ALERT_COLOR[d.alert] || '#E87516';
    const riskColor  = RISK_STYLE[d.riskLevel]?.fill || '#16A34A';

    const popupContent = `
      <div style="font-family:'Inter',sans-serif;min-width:220px;max-width:260px;font-size:12px;line-height:1.5">
        <div style="background:${riskColor};color:#fff;padding:8px 10px;margin:-12px -12px 10px -12px;border-radius:4px 4px 0 0">
          <div style="font-weight:800;font-size:14px">${d.district}</div>
          <div style="font-size:11px;opacity:0.9">${d.state}</div>
        </div>
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px">
          <span style="background:${alertColor};color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px">${d.alert}</span>
          <span style="font-size:11px;color:#6B7280">Risk: <strong style="color:${riskColor}">${d.riskLevel}</strong></span>
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:11.5px">
          <tr style="border-bottom:1px solid #E5E7EB">
            <td style="padding:3px 0;color:#6B7280">⛈ Thunderstorm</td>
            <td style="text-align:right;font-weight:700;color:#7E22CE">${d.thunderstormProbability}%</td>
          </tr>
          <tr style="border-bottom:1px solid #E5E7EB">
            <td style="padding:3px 0;color:#6B7280">🌧 Cloudburst</td>
            <td style="text-align:right;font-weight:700;color:#DC2626">${d.cloudburstProbability}%</td>
          </tr>
          <tr style="border-bottom:1px solid #E5E7EB">
            <td style="padding:3px 0;color:#6B7280">🌊 Flash Flood</td>
            <td style="text-align:right;font-weight:700;color:#0284C7">${d.flashFloodProbability}%</td>
          </tr>
          <tr style="border-bottom:1px solid #E5E7EB">
            <td style="padding:3px 0;color:#6B7280">⏱ Lead Time</td>
            <td style="text-align:right;font-weight:700;color:#1D4ED8">${d.expectedLeadTime}</td>
          </tr>
          <tr style="border-bottom:1px solid #E5E7EB">
            <td style="padding:3px 0;color:#6B7280">🌧 24h Rainfall</td>
            <td style="text-align:right;font-weight:700">${d.rainfall24h}</td>
          </tr>
          <tr>
            <td style="padding:3px 0;color:#6B7280">🏞 Key Rivers</td>
            <td style="text-align:right;color:#374151">${d.rivers}</td>
          </tr>
        </table>
      </div>
    `;

    e.target.bindPopup(popupContent, { maxWidth: 280 }).openPopup();
  }, [setSelectedDistrict]);

  const onEachFeature = useCallback((feature, layer) => {
    layer.on({
      mouseover: highlightFeature,
      mouseout:  resetHighlight,
      click:     onFeatureClick,
    });
  }, [highlightFeature, resetHighlight, onFeatureClick]);

  // ── Tile URL ────────────────────────────────────────────────────
  const tileUrl = {
    satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    osm:       'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    topo:      'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
  }[basemap] || 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';

  // Current overlay label
  const activeLayerLabel = activeOverlay
    ? MAP_LAYER_DEFINITIONS.find((l) => l.key === activeOverlay)?.label
    : (mapLayers.risk ? 'Risk Level' : null);

  return (
    <div className="bg-white border border-[#E2E2E2] rounded shadow-sm overflow-hidden flex flex-col h-full">
      {/* ── Card Header ──────────────────────────────────────────── */}
      <div className="px-3 py-2 border-b border-[#E2E2E2] flex items-center justify-between bg-white shrink-0">
        <div className="flex items-center gap-2">
          <h2 className="text-[13px] font-bold text-[#172033] m-0 leading-none flex items-center gap-1.5">
            <span className="w-1 h-3.5 bg-[#E87516] rounded-sm inline-block"></span>
            <span>Live Risk Map</span>
          </h2>
          <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded text-emerald-700 bg-emerald-50 border border-emerald-200">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse inline-block"></span>
            LIVE
          </span>
          {activeLayerLabel && (
            <span className="text-[10px] font-semibold text-[#C85D00] bg-[#FFF1DD] border border-orange-200 px-2 py-0.5 rounded">
              Layer: {activeLayerLabel}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-slate-500 font-medium">
            {region.name}
          </span>
          <span className="text-[11px] font-semibold text-slate-600 bg-slate-100 px-2 py-0.5 rounded border border-slate-300">
            {getCurrentISTTime()}
          </span>
        </div>
      </div>

      {/* ── Map Viewport ─────────────────────────────────────────── */}
      <div className={`relative w-full ${height} flex-1 overflow-hidden`}>
        <MapContainer
          center={mapCenter}
          zoom={mapZoom}
          zoomControl={false}
          attributionControl={false}
          className="w-full h-full"
          ref={mapRef}
          style={{ background: '#162738' }}
        >
          <MapViewController center={mapCenter} zoom={mapZoom} />
          <TileLayer url={tileUrl} maxZoom={18} />
          <GeoJSON
            key={`${selectedRegion}-${JSON.stringify(mapLayers)}-${showBorders}-${basemap}`}
            ref={geoJsonRef}
            data={filteredGeoJSON}
            style={getDistrictStyle}
            onEachFeature={onEachFeature}
          />
          <PlaceMarkersController selectedRegion={selectedRegion} />
        </MapContainer>

        {/* ── Zoom Controls (Top-Left) ───────────────────────────── */}
        {showControls && (
          <div className="absolute top-3 left-3 z-[500] flex flex-col gap-1.5">
            <button title="Zoom In" onClick={() => mapRef.current?.zoomIn()}
              className="w-7 h-7 bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 rounded shadow flex items-center justify-center">
              <Plus className="w-3.5 h-3.5" />
            </button>
            <button title="Zoom Out" onClick={() => mapRef.current?.zoomOut()}
              className="w-7 h-7 bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 rounded shadow flex items-center justify-center">
              <Minus className="w-3.5 h-3.5" />
            </button>
            <button title="Reset View" onClick={() => mapRef.current?.setView(mapCenter, mapZoom, { animate: true })}
              className="w-7 h-7 bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 rounded shadow flex items-center justify-center">
              <Crosshair className="w-3.5 h-3.5" />
            </button>

            {/* Layer Panel Toggle */}
            <div className="relative">
              <button title="Layer Controls" onClick={() => setShowLayerPanel((v) => !v)}
                className={`w-7 h-7 border rounded shadow flex items-center justify-center transition-colors ${
                  showLayerPanel ? 'bg-[#E87516] text-white border-[#E87516]' : 'bg-white hover:bg-[#FFF7EA] text-slate-700 border-slate-300'
                }`}>
                <LayersIcon className="w-3.5 h-3.5" />
              </button>

              {showLayerPanel && (
                <div className="absolute left-9 top-0 bg-white border border-[#E2E2E2] shadow-xl rounded z-[600] w-56 text-xs">
                  {/* Base Map Section */}
                  <div className="px-3 py-2 border-b border-slate-100">
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-1.5 tracking-wide">Base Map</p>
                    {[
                      { id: 'satellite', label: 'Satellite / Imagery' },
                      { id: 'topo',      label: 'Topographic' },
                      { id: 'osm',       label: 'OpenStreetMap' },
                    ].map(({ id, label }) => (
                      <label key={id} className="flex items-center gap-2 cursor-pointer py-0.5">
                        <input type="radio" name="basemap" value={id} checked={basemap === id}
                          onChange={() => setBasemap(id)} className="accent-[#E87516]" />
                        <span className="text-slate-700">{label}</span>
                      </label>
                    ))}
                  </div>

                  {/* Data Layers Section */}
                  <div className="px-3 py-2 border-b border-slate-100">
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-1.5 tracking-wide">Data Layers</p>
                    {MAP_LAYER_DEFINITIONS.map(({ key, label, icon }) => (
                      <label key={key} className="flex items-center gap-2 cursor-pointer py-0.5">
                        <input type="checkbox" checked={!!mapLayers[key]}
                          onChange={() => toggleMapLayer(key)} className="rounded accent-[#E87516]" />
                        <LayerIcon iconKey={icon} className="w-3.5 h-3.5 text-slate-500" />
                        <span className="text-slate-700">{label}</span>
                      </label>
                    ))}
                  </div>

                  {/* Borders Toggle */}
                  <div className="px-3 py-2">
                    <label className="flex items-center gap-2 cursor-pointer py-0.5">
                      <input type="checkbox" checked={showBorders}
                        onChange={toggleBorders} className="rounded accent-[#E87516]" />
                      <span className="text-slate-700">District Boundaries</span>
                    </label>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Map Layer Selector (Top-Right) ────────────────────── */}
        {showControls && <MapLayerControl />}

        {/* ── Hover Tooltip (Top-Right, offset) ──────────────────── */}
        {hoveredDistrict && (
          <div className="absolute top-14 right-3 z-[490] bg-white/96 border border-slate-300 rounded shadow-lg px-3 py-2 min-w-[170px] pointer-events-none">
            <div className="text-[12px] font-extrabold text-slate-900">{hoveredDistrict.district}</div>
            <div className="text-[10.5px] text-slate-500">{hoveredDistrict.state}</div>
            <div className="text-[11px] font-bold mt-1 px-1.5 py-0.5 rounded inline-block text-white"
              style={{ background: RISK_STYLE[hoveredDistrict.riskLevel]?.fill || '#16A34A' }}>
              {hoveredDistrict.riskLevel}
            </div>
            <div className="text-[10.5px] text-slate-500 mt-1 flex items-center gap-1">
              <Info className="w-3 h-3" /> Click for full details
            </div>
          </div>
        )}

        {/* ── Risk Legend (Bottom-Left) ──────────────────────────── */}
        {mapLayers.risk && (
          <div className="absolute bottom-3 left-3 z-[500] bg-white/95 border border-slate-300 rounded shadow px-2.5 py-2 flex flex-col gap-1">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wide block mb-0.5">Risk Level</span>
            {LEGEND_ITEMS.map(({ color, label }) => (
              <div key={label} className="flex items-center gap-1.5">
                <span className="w-4 h-3 rounded-sm shrink-0" style={{ backgroundColor: color, opacity: 0.75 }} />
                <span className="text-[11px] text-slate-700 font-medium">{label}</span>
              </div>
            ))}
          </div>
        )}

        {/* ── District Summary (Bottom-Right) ───────────────────── */}
        <div className="absolute bottom-3 right-3 z-[500] bg-white/95 border border-slate-300 rounded shadow px-2.5 py-2">
          <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wide block mb-1">
            District Summary
          </span>
          {(['EXTREME', 'VERY HIGH', 'HIGH', 'MODERATE', 'LOW']).map((level) => {
            const regionFilter = selectedRegion === 'uk-hp' ? null
              : selectedRegion === 'uk' ? 'Uttarakhand' : 'Himachal Pradesh';
            const count = DISTRICT_RISK_DATA.filter((d) =>
              d.riskLevel === level && (!regionFilter || d.state === regionFilter)
            ).length;
            const { fill } = RISK_STYLE[level];
            return (
              <div key={level} className="flex items-center justify-between gap-3 text-[11px]">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: fill }} />
                  <span className="text-slate-600 font-medium capitalize">{level.toLowerCase()}</span>
                </div>
                <span className="font-bold text-slate-900">{count}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default RiskMap;
