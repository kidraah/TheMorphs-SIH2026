import React, { useState } from 'react';
import RiskMap from '../components/RiskMap';
import { DISTRICT_RISK_DATA } from '../data/districtData';
import { Layers, MapPin, Eye, ShieldAlert, Droplets, Mountain } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

export const LiveMapPage = () => {
  const { mapLayers, toggleMapLayer, selectedDistrict: globalSelectedDistrict, setSelectedDistrict } = useAppStore();
  const selectedDistrict = globalSelectedDistrict || DISTRICT_RISK_DATA[0];

  return (
    <div className="space-y-4">
      {/* Page Title Header */}
      <div className="bg-white border border-slate-200 rounded p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-gov-sm">
        <div>
          <h2 className="text-base font-bold text-[#0A3871] m-0 flex items-center gap-2">
            <Layers className="w-5 h-5 text-blue-700" />
            Geospatial Convective Risk & Nowcasting Portal
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 m-0">
            Real-time multi-sensor radar, satellite, and terrain isohyet nowcast visualization
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="px-2.5 py-1 bg-emerald-50 text-emerald-800 border border-emerald-300 font-bold rounded">
            Live Stream Active
          </span>
          <span className="px-2.5 py-1 bg-slate-100 text-slate-700 border border-slate-300 font-medium rounded">
            Spatial Res: 1km²
          </span>
        </div>
      </div>

      {/* Main Map + District Inspector Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Expanded Map Viewport */}
        <div className="lg:col-span-8 flex flex-col">
          <RiskMap height="h-[560px]" />
        </div>

        {/* District Detail & Layer Control Column */}
        <div className="lg:col-span-4 space-y-4">
          {/* Layer Control Card */}
          <div className="bg-white border border-slate-200 rounded p-3.5 shadow-gov-sm">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2.5 border-b pb-1.5 flex items-center gap-1.5">
              <Eye className="w-4 h-4 text-slate-600" />
              Active GIS Layer Controls
            </h3>
            <div className="space-y-2 text-xs">
              <label className="flex items-center justify-between p-2 rounded bg-slate-50 hover:bg-slate-100 cursor-pointer border border-slate-200">
                <span className="font-semibold text-slate-800">Satellite Topography Base</span>
                <input
                  type="checkbox"
                  checked={mapLayers.satellite}
                  onChange={() => toggleMapLayer('satellite')}
                  className="rounded text-blue-600 w-4 h-4"
                />
              </label>
              <label className="flex items-center justify-between p-2 rounded bg-slate-50 hover:bg-slate-100 cursor-pointer border border-slate-200">
                <span className="font-semibold text-slate-800">Convective Risk Contours</span>
                <input
                  type="checkbox"
                  checked={mapLayers.riskContours}
                  onChange={() => toggleMapLayer('riskContours')}
                  className="rounded text-blue-600 w-4 h-4"
                />
              </label>
              <label className="flex items-center justify-between p-2 rounded bg-slate-50 hover:bg-slate-100 cursor-pointer border border-slate-200">
                <span className="font-semibold text-slate-800">District Polygons & Labels</span>
                <input
                  type="checkbox"
                  checked={mapLayers.districtBorders}
                  onChange={() => toggleMapLayer('districtBorders')}
                  className="rounded text-blue-600 w-4 h-4"
                />
              </label>
            </div>
          </div>

          {/* District Inspector Card */}
          <div className="bg-white border border-slate-200 rounded p-3.5 shadow-gov-sm">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2.5 border-b pb-1.5 flex items-center gap-1.5">
              <MapPin className="w-4 h-4 text-red-600" />
              District Focus Inspector
            </h3>
            
            <div className="mb-3">
              <label className="text-[11px] font-semibold text-slate-500 block mb-1">
                Select District to Inspect:
              </label>
              <select
                value={selectedDistrict.id}
                onChange={(e) => {
                  const dist = DISTRICT_RISK_DATA.find((d) => d.id === e.target.value);
                  if (dist) setSelectedDistrict(dist);
                }}
                className="w-full text-xs font-medium text-slate-800 bg-slate-50 border border-slate-300 rounded p-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                {DISTRICT_RISK_DATA.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.state}) - {d.riskLevel}
                  </option>
                ))}
              </select>
            </div>

            {/* Selected District Metrics */}
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between p-2 bg-red-50/80 border border-red-200 rounded">
                <span className="font-bold text-red-900 flex items-center gap-1.5">
                  <ShieldAlert className="w-4 h-4 text-red-600" />
                  Risk Classification
                </span>
                <span className="font-extrabold text-red-700">{selectedDistrict.riskLevel}</span>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="p-2 bg-slate-50 border border-slate-200 rounded">
                  <span className="text-[10px] text-slate-500 font-bold block uppercase">Cloudburst Prob</span>
                  <span className="text-sm font-extrabold text-purple-700">{selectedDistrict.cloudburstProb}%</span>
                </div>
                <div className="p-2 bg-slate-50 border border-slate-200 rounded">
                  <span className="text-[10px] text-slate-500 font-bold block uppercase">Flash Flood Prob</span>
                  <span className="text-sm font-extrabold text-blue-700">{selectedDistrict.flashFloodProb}%</span>
                </div>
              </div>

              <div className="p-2 bg-slate-50 border border-slate-200 rounded space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-500 flex items-center gap-1">
                    <Droplets className="w-3.5 h-3.5 text-blue-500" />
                    24h Rainfall:
                  </span>
                  <span className="font-bold text-slate-800">{selectedDistrict.rainfall24h}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 flex items-center gap-1">
                    <Mountain className="w-3.5 h-3.5 text-emerald-600" />
                    Vulnerable Rivers:
                  </span>
                  <span className="font-semibold text-slate-800">{selectedDistrict.rivers}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveMapPage;
