import React from 'react';
import DataSources from '../components/DataSources';
import { Database, Activity, Map, Radio, RefreshCw } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

export const DataLayersPage = () => {
  const { dataSources } = useAppStore();

  const telemetryFeeds = [
    {
      name: 'INSAT-3DR TIR1 (10.8 µm) Thermal Infrared',
      type: 'Satellite Radiance',
      resolution: '4 km / 15 min',
      provider: 'ISRO / IMD SAC',
      status: 'Online',
      latency: '2.4 min',
    },
    {
      name: 'INSAT-3D Water Vapor Band (6.5 - 7.1 µm)',
      type: 'Mid-Troposphere Moisture',
      resolution: '8 km / 15 min',
      provider: 'ISRO / IMD',
      status: 'Online',
      latency: '2.1 min',
    },
    {
      name: 'Doppler Weather Radar (DWR) Reflectivity (Z)',
      type: 'C-Band / S-Band Radar',
      resolution: '250 m / 10 min',
      provider: 'IMD Radar Network (Mukteshwar / Patiala)',
      status: 'Online',
      latency: '1.2 min',
    },
    {
      name: 'IMDAA High-Resolution Atmospheric Reanalysis',
      type: 'Reanalysis Grids',
      resolution: '12 km / Hourly',
      provider: 'NCMRWF / MoES',
      status: 'Online',
      latency: '14 min',
    },
    {
      name: 'Quantitative Precipitation Estimation (QPE)',
      type: 'Hydro-Estimator Algorithm',
      resolution: '1 km / 15 min',
      provider: 'IMD Hydromet Division',
      status: 'Online',
      latency: '3.8 min',
    },
    {
      name: 'High-Res Digital Elevation Model (SRTM/CartoDEM)',
      type: 'Topographic Slope & Aspect',
      resolution: '30 m / Static',
      provider: 'ISRO Bhuvan / Survey of India',
      status: 'Active',
      latency: 'Static Mesh',
    },
  ];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-white border border-[#E2E2E2] rounded p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-gov-sm">
        <div>
          <h2 className="text-base font-bold text-[#172033] m-0 flex items-center gap-2">
            <span className="w-1.5 h-4.5 bg-[#E87516] rounded-sm inline-block"></span>
            <Database className="w-5 h-5 text-[#E87516]" />
            Ingested Observational Feeds & Geospatial Data Layers
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 m-0">
            Real-time telemetry ingestion pipelines supporting the AI Hyper-Local Nowcasting Engine
          </p>
        </div>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="px-3 py-1.5 bg-slate-100 hover:bg-[#FFF7EA] text-[#172033] text-xs font-semibold rounded flex items-center gap-1.5 border border-slate-300 shadow-sm transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Poll Telemetry Pipelines
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mb-4">
        <div className="bg-white border border-[#E2E2E2] rounded p-4 shadow-gov-sm flex items-start gap-3">
          <div className="p-2 bg-emerald-100 text-emerald-700 rounded-lg shrink-0">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <p className="text-[11px] font-bold text-slate-500 uppercase">System Health</p>
            <p className="text-lg font-extrabold text-slate-800">98.5% Uptime</p>
          </div>
        </div>
        <div className="bg-white border border-[#E2E2E2] rounded p-4 shadow-gov-sm flex items-start gap-3">
          <div className="p-2 bg-blue-100 text-blue-700 rounded-lg shrink-0">
            <Radio className="w-5 h-5" />
          </div>
          <div>
            <p className="text-[11px] font-bold text-slate-500 uppercase">Active Streams</p>
            <p className="text-lg font-extrabold text-slate-800">12 / 14 Available</p>
          </div>
        </div>
        <div className="bg-white border border-[#E2E2E2] rounded p-4 shadow-gov-sm flex items-start gap-3">
          <div className="p-2 bg-purple-100 text-purple-700 rounded-lg shrink-0">
            <Map className="w-5 h-5" />
          </div>
          <div>
            <p className="text-[11px] font-bold text-slate-500 uppercase">Geospatial Nodes</p>
            <p className="text-lg font-extrabold text-slate-800">4,281 Covered</p>
          </div>
        </div>
      </div>

      {/* Main component */}
      <div className="bg-white border border-[#E2E2E2] rounded shadow-gov-sm min-h-[500px] p-4">
        <DataSources sources={dataSources || []} />
      </div>

      {/* Telemetry Table */}
      <div className="bg-white border border-[#E2E2E2] rounded shadow-gov-sm overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="bg-slate-100/90 text-slate-700 border-b border-slate-300 font-bold uppercase text-[10.5px]">
              <th className="p-2.5">Data Stream / Sensor</th>
              <th className="p-2.5">Sensor Domain</th>
              <th className="p-2.5">Spatial & Temporal Res</th>
              <th className="p-2.5">Providing Agency</th>
              <th className="p-2.5">Ingestion Status</th>
              <th className="p-2.5">Stream Latency</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200">
            {telemetryFeeds.map((feed, idx) => (
              <tr key={idx} className="hover:bg-slate-50 transition-colors">
                <td className="p-2.5 font-bold text-[#172033]">{feed.name}</td>
                <td className="p-2.5 text-slate-600">{feed.type}</td>
                <td className="p-2.5 font-mono text-slate-700">{feed.resolution}</td>
                <td className="p-2.5 text-slate-700">{feed.provider}</td>
                <td className="p-2.5">
                  <span className="px-2 py-0.5 bg-emerald-50 text-emerald-800 border border-emerald-200 rounded font-bold text-[10.5px]">
                    {feed.status}
                  </span>
                </td>
                <td className="p-2.5 font-mono text-slate-500">{feed.latency}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DataLayersPage;
