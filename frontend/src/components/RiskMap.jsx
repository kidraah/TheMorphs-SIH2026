import React, { useState, useCallback, useMemo, useEffect } from 'react';
import DeckGL from '@deck.gl/react';
import Map, { Source, Layer } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { GeoJsonLayer, BitmapLayer } from '@deck.gl/layers';
import { TripsLayer } from '@deck.gl/geo-layers';
import { MaskExtension } from '@deck.gl/extensions';
import {
  Plus, Minus, Crosshair, Layers as LayersIcon, Info,
  ShieldAlert, CloudRain, Droplet, Thermometer, Zap, Wind, Mountain,
} from 'lucide-react';
import { HIMALAYAN_DISTRICTS_GEOJSON } from '../geojson/himalayanDistricts';
import { PLACE_LABELS } from '../data/placeLabels';
import { MAP_LAYER_DEFINITIONS } from '../data/mockData';
import { MapLayerControl } from './MapLayerControl';
import { useAppStore } from '../store/useAppStore';
import { getCurrentISTTime } from '../utils/timeFormatter';

// ─── Constants ────────────────────────────────────────────────────

const RISK_STYLE = {
  LOW:         { fill: [22, 163, 74], opacity: 0.32, border: [21, 128, 61] },
  MODERATE:    { fill: [202, 138, 4], opacity: 0.42, border: [161, 98, 7] },
  HIGH:        { fill: [234, 88, 12], opacity: 0.50, border: [194, 65, 12] },
  'VERY HIGH': { fill: [220, 38, 38], opacity: 0.56, border: [185, 28, 28] },
  EXTREME:     { fill: [126, 34, 206], opacity: 0.62, border: [107, 33, 168] },
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

const OVERLAY_COLORS = {
  risk:   null,
  precip: [14, 165, 233],
  iwv:    [99, 102, 241],
  ctt:    [139, 92, 246],
  cape:   [245, 158, 11],
  wind:   [20, 184, 166],
  dem:    [120, 113, 108],
};

const LayerIcon = ({ iconKey, className = 'w-3.5 h-3.5' }) => {
  switch (iconKey) {
    case 'shield':      return <ShieldAlert className={className} />;
    case 'cloud-rain':  return <CloudRain className={className} />;
    case 'droplet':     return <Droplet className={className} />;
    case 'thermometer': return <Thermometer className={className} />;
    case 'zap':         return <Zap className={className} />;
    case 'wind':        return <Wind className={className} />;
    case 'mountain':    return <Mountain className={className} />;
    default:            return <LayersIcon className={className} />;
  }
};

// No token needed for MapLibre + Free tiles

// ─── Main Component ───────────────────────────────────────────────

export const RiskMap = ({ height = 'h-[520px]', showControls = true }) => {
  const {
    selectedRegion, regions,
    mapLayers, toggleMapLayer,
    basemap, setBasemap,
    showBorders, toggleBorders,
    setSelectedDistrict,
    districtRiskData,
    geoHeatmap,
    geoTrajectories,
  } = useAppStore();

  const RISK_INDEX = useMemo(() => Object.fromEntries(
    (districtRiskData || []).map((d) => [d.district.toLowerCase(), d])
  ), [districtRiskData]);

  const [showLayerPanel, setShowLayerPanel] = useState(false);
  const [hoverInfo, setHoverInfo] = useState(null);
  
  // Animation frame for Flood Polygons
  const [frameIndex, setFrameIndex] = useState(0);

  useEffect(() => {
    let animation;
    let lastTime = 0;
    const animate = (timestamp) => {
      if (timestamp - lastTime > 200) { // Update frame every 200ms
        if (geoTrajectories && geoTrajectories.length > 0) {
          setFrameIndex(prev => (prev + 1) % geoTrajectories.length);
        }
        lastTime = timestamp;
      }
      animation = window.requestAnimationFrame(animate);
    };
    animate(0);
    return () => window.cancelAnimationFrame(animation);
  }, [geoTrajectories]);

  const currentFrameGeoJSON = geoTrajectories && geoTrajectories[frameIndex] ? geoTrajectories[frameIndex] : null;

  const region = regions.find((r) => r.id === selectedRegion) || regions[0];
  
  // DeckGL ViewState
  const [viewState, setViewState] = useState({
    longitude: region.center[1],
    latitude: region.center[0],
    zoom: region.zoom - 0.5,
    pitch: 45,
    bearing: 0,
    transitionDuration: 1000
  });

  // Sync ViewState if region changes externally
  useEffect(() => {
    setViewState(prev => ({
      ...prev,
      longitude: region.center[1],
      latitude: region.center[0],
      zoom: region.zoom - 0.5,
      transitionDuration: 1000
    }));
  }, [region.id]);

  const activeOverlay = useMemo(() => {
    for (const key of Object.keys(mapLayers)) {
      if (key !== 'risk' && mapLayers[key]) return key;
    }
    return null;
  }, [mapLayers]);

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

  const getDistrictFillColor = useCallback((feature) => {
    const key = (feature.properties.districtKey || '').toLowerCase();
    const data = RISK_INDEX[key];

    if (mapLayers.risk) {
      const { fill, opacity } = (data && RISK_STYLE[data.riskLevel]) || RISK_STYLE.LOW;
      return [...fill, opacity * 255];
    }

    if (activeOverlay && OVERLAY_COLORS[activeOverlay]) {
      const overlayColor = OVERLAY_COLORS[activeOverlay];
      let intensity = 0.2;
      if (data) {
        const avg = (data.thunderstormProbability + data.cloudburstProbability + data.flashFloodProbability) / 3;
        intensity = 0.15 + (avg / 100) * 0.50;
      }
      return [...overlayColor, intensity * 255];
    }

    return [0, 0, 0, 0];
  }, [mapLayers, activeOverlay, RISK_INDEX]);

  const getDistrictLineColor = useCallback((feature) => {
    if (!showBorders) return [0, 0, 0, 0];
    
    const key = (feature.properties.districtKey || '').toLowerCase();
    const data = RISK_INDEX[key];

    if (mapLayers.risk) {
      const { border } = (data && RISK_STYLE[data.riskLevel]) || RISK_STYLE.LOW;
      return [...border, 255];
    }

    if (activeOverlay && OVERLAY_COLORS[activeOverlay]) {
      return [...OVERLAY_COLORS[activeOverlay], 255];
    }

    return [148, 163, 184, 255]; // slate-400
  }, [mapLayers, showBorders, activeOverlay, RISK_INDEX]);

  const layers = [];

  // Mask layer for clipping the heatmap exactly to district polygons
  layers.push(
    new GeoJsonLayer({
      id: 'districts-mask-layer',
      data: filteredGeoJSON,
      operation: 'mask'
    })
  );

  // 1. Bitmap Heatmap Layer
  if (mapLayers['imdaa-risk'] && geoHeatmap) {
    layers.push(
      new BitmapLayer({
        id: 'heatmap-layer',
        bounds: [geoHeatmap.bounds[0][1], geoHeatmap.bounds[0][0], geoHeatmap.bounds[1][1], geoHeatmap.bounds[1][0]], // [minX, minY, maxX, maxY]
        image: geoHeatmap.image,
        opacity: geoHeatmap.opacity,
        transparentColor: [0, 0, 0, 0],
        extensions: [new MaskExtension()],
        maskId: 'districts-mask-layer'
      })
    );
  }

  // 2. Trajectories (TripsLayer) has been replaced by MapLibre native Source/Layer in the render block

  // 3. District GeoJSON Layer
  layers.push(
    new GeoJsonLayer({
      id: 'districts-layer',
      data: filteredGeoJSON,
      pickable: true,
      stroked: true,
      filled: true,
      extruded: false,
      lineWidthMinPixels: showBorders ? 1.5 : 0,
      getFillColor: getDistrictFillColor,
      getLineColor: getDistrictLineColor,
      onHover: info => setHoverInfo(info),
      onClick: info => {
        const key = (info.object?.properties.districtKey || '').toLowerCase();
        const d = RISK_INDEX[key];
        if (d) setSelectedDistrict(d);
      },
      updateTriggers: {
        getFillColor: [mapLayers, activeOverlay, RISK_INDEX],
        getLineColor: [mapLayers, showBorders, activeOverlay, RISK_INDEX]
      }
    })
  );

  // Free open source raster styles
  const mapStyle = {
    version: 8,
    sources: {
      'raster-tiles': {
        type: 'raster',
        tiles: [
          basemap === 'satellite'
            ? 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
            : basemap === 'topo'
            ? 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}'
            : 'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'
        ],
        tileSize: 256,
        attribution: 'Map data © OpenStreetMap contributors, ESRI, Carto'
      }
    },
    layers: [{
      id: 'base-layer',
      type: 'raster',
      source: 'raster-tiles',
      minzoom: 0,
      maxzoom: 19
    }]
  };

  const activeLayerLabel = activeOverlay
    ? MAP_LAYER_DEFINITIONS.find((l) => l.key === activeOverlay)?.label
    : (mapLayers.risk ? 'Risk Level' : null);

  return (
    <div className={"bg-white border border-[#E2E2E2] rounded shadow-sm overflow-hidden flex flex-col " + height}>
      {/* ── Card Header ──────────────────────────────────────────── */}
      <div className="px-3 py-2 border-b border-[#E2E2E2] flex items-center justify-between bg-white shrink-0 z-[100]">
        <div className="flex items-center gap-2">
          <h2 className="text-[13px] font-bold text-[#172033] m-0 leading-none flex items-center gap-1.5">
            <span className="w-1 h-3.5 bg-[#E87516] rounded-sm inline-block"></span>
            <span>Live Risk Map (3D Terrain)</span>
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

      {/* ── DeckGL Viewport ─────────────────────────────────────────── */}
      <div className="relative w-full flex-1 overflow-hidden">
        <DeckGL
          layers={layers}
          viewState={viewState}
          onViewStateChange={e => setViewState(e.viewState)}
          controller={true}
          getTooltip={({object}) => {
             if (!object) return null;
             const key = (object.properties.districtKey || '').toLowerCase();
             const d = RISK_INDEX[key];
             if (!d) return object.properties.districtKey;
             
             return {
                 html: `
                    <div style="font-family:'Inter',sans-serif;font-size:12px;padding:4px;min-width:140px;">
                        <div style="font-weight:800;font-size:14px;color:#111;">${d.district}</div>
                        <div style="font-size:10.5px;color:#666;margin-bottom:6px;">${d.state}</div>
                        <div style="font-weight:700;padding:2px 6px;border-radius:3px;display:inline-block;
                            background:${RISK_STYLE[d.riskLevel]?.fill ? 'rgba('+RISK_STYLE[d.riskLevel].fill.join(',')+',0.2)' : '#eee'};
                            color:${RISK_STYLE[d.riskLevel]?.border ? 'rgb('+RISK_STYLE[d.riskLevel].border.join(',')+')' : '#333'}">
                            ${d.riskLevel} RISK
                        </div>
                    </div>
                 `,
                 style: { backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '4px', color: '#333', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }
             };
          }}
        >
          <Map
            mapStyle={mapStyle}
            terrain={{source: 'aws-terrain', exaggeration: 1.5}}
          >
            <source
              id="aws-terrain"
              type="raster-dem"
              tiles={['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png']}
              encoding="terrarium"
              tileSize={256}
              maxzoom={14}
            />
            {mapLayers['qpe-rain'] && currentFrameGeoJSON && (
              <Source id="flood-data" type="geojson" data={currentFrameGeoJSON}>
                <Layer 
                  id="flood-fill" 
                  type="fill" 
                  paint={{
                    'fill-color': '#3b82f6', 
                    'fill-opacity': 0.5
                  }} 
                />
              </Source>
            )}
          </Map>
        </DeckGL>

        {/* ── Zoom Controls (Top-Left) ───────────────────────────── */}
        {showControls && (
          <div className="absolute top-3 left-3 z-[500] flex flex-col gap-1.5 pointer-events-auto">
            <button title="Zoom In" onClick={() => setViewState(v => ({...v, zoom: v.zoom + 1, transitionDuration: 300}))}
              className="w-7 h-7 bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 rounded shadow flex items-center justify-center">
              <Plus className="w-3.5 h-3.5" />
            </button>
            <button title="Zoom Out" onClick={() => setViewState(v => ({...v, zoom: v.zoom - 1, transitionDuration: 300}))}
              className="w-7 h-7 bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 rounded shadow flex items-center justify-center">
              <Minus className="w-3.5 h-3.5" />
            </button>
            <button title="Reset View" onClick={() => setViewState(v => ({...v, longitude: region.center[1], latitude: region.center[0], zoom: region.zoom - 0.5, pitch: 45, bearing: 0, transitionDuration: 1000}))}
              className="w-7 h-7 bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 rounded shadow flex items-center justify-center">
              <Crosshair className="w-3.5 h-3.5" />
            </button>

            {/* Layer Panel Toggle */}
            <div className="relative pointer-events-auto">
              <button title="Layer Controls" onClick={() => setShowLayerPanel((v) => !v)}
                className={`w-7 h-7 border rounded shadow flex items-center justify-center transition-colors ${
                  showLayerPanel ? 'bg-[#E87516] text-white border-[#E87516]' : 'bg-white hover:bg-[#FFF7EA] text-slate-700 border-slate-300'
                }`}>
                <LayersIcon className="w-3.5 h-3.5" />
              </button>

              {showLayerPanel && (
                <div className="absolute left-9 top-0 bg-white border border-[#E2E2E2] shadow-xl rounded z-[600] w-56 text-xs">
                  <div className="px-3 py-2 border-b border-slate-100">
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-1.5 tracking-wide">Base Map</p>
                    {[
                      { id: 'satellite', label: 'Satellite / Imagery' },
                      { id: 'topo',      label: 'Topographic' }
                    ].map(({ id, label }) => (
                      <label key={id} className="flex items-center gap-2 cursor-pointer py-0.5">
                        <input type="radio" name="basemap" value={id} checked={basemap === id}
                          onChange={() => setBasemap(id)} className="accent-[#E87516]" />
                        <span className="text-slate-700">{label}</span>
                      </label>
                    ))}
                  </div>

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
        {showControls && (
          <div className="absolute top-3 right-3 z-[500] pointer-events-auto">
             <MapLayerControl />
          </div>
        )}

        {/* ── Risk Legend (Bottom-Left) ──────────────────────────── */}
        {mapLayers.risk && (
          <div className="absolute bottom-3 left-3 z-[500] bg-white/95 border border-slate-300 rounded shadow px-2.5 py-2 flex flex-col gap-1 pointer-events-auto">
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
        <div className="absolute bottom-3 right-3 z-[500] bg-white/95 border border-slate-300 rounded shadow px-2.5 py-2 pointer-events-auto">
          <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wide block mb-1">
            District Summary
          </span>
          {(['EXTREME', 'VERY HIGH', 'HIGH', 'MODERATE', 'LOW']).map((level) => {
            const regionFilter = selectedRegion === 'uk-hp' ? null
              : selectedRegion === 'uk' ? 'Uttarakhand' : 'Himachal Pradesh';
            const count = (districtRiskData || []).filter((d) =>
              d.riskLevel === level && (!regionFilter || d.state === regionFilter)
            ).length;
            
            if (count === 0) return null;
            
            const fill = RISK_STYLE[level].fill;
            return (
              <div key={level} className="flex items-center justify-between gap-3 text-[11px]">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: `rgb(${fill.join(',')})` }} />
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
