import React, { useState } from 'react';
import { Settings, Sliders, BellRing, Save, ShieldCheck } from 'lucide-react';

export const SettingsPage = () => {
  const [cloudburstThreshold, setCloudburstThreshold] = useState(80);
  const [flashFloodThreshold, setFlashFloodThreshold] = useState(70);
  const [refreshInterval, setRefreshInterval] = useState(15);
  const [capAlerts, setCapAlerts] = useState(true);
  const [saved, setSaved] = useState(false);

  const handleSave = (e) => {
    e.preventDefault();
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-white border border-slate-200 rounded p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-gov-sm">
        <div>
          <h2 className="text-base font-bold text-[#0A3871] m-0 flex items-center gap-2">
            <Settings className="w-5 h-5 text-blue-700" />
            System Configuration & Alert Dispatch Settings
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 m-0">
            Threshold tuning for AI convective risk triggers and NDMA CAP protocol feeds
          </p>
        </div>
      </div>

      <form onSubmit={handleSave} className="space-y-4">
        {/* Risk Thresholds Card */}
        <div className="bg-white border border-slate-200 rounded p-4 shadow-gov-sm space-y-3">
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider border-b pb-2 flex items-center gap-1.5">
            <Sliders className="w-4 h-4 text-slate-700" />
            Operational Trigger Thresholds
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <div className="flex justify-between font-semibold text-slate-700 mb-1">
                <span>Cloudburst Red Alert Probability Threshold:</span>
                <span className="font-bold text-red-600">{cloudburstThreshold}%</span>
              </div>
              <input
                type="range"
                min="50"
                max="95"
                value={cloudburstThreshold}
                onChange={(e) => setCloudburstThreshold(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-red-600"
              />
              <span className="text-[11px] text-slate-500 block mt-1">
                Triggers urgent red warnings to SDMA when convective AI score exceeds this value.
              </span>
            </div>

            <div>
              <div className="flex justify-between font-semibold text-slate-700 mb-1">
                <span>Flash Flood Inundation Threshold:</span>
                <span className="font-bold text-blue-600">{flashFloodThreshold}%</span>
              </div>
              <input
                type="range"
                min="40"
                max="90"
                value={flashFloodThreshold}
                onChange={(e) => setFlashFloodThreshold(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
              <span className="text-[11px] text-slate-500 block mt-1">
                Calculated in conjunction with high-res digital elevation models and river cross-sections.
              </span>
            </div>
          </div>
        </div>

        {/* Integration Feeds Card */}
        <div className="bg-white border border-slate-200 rounded p-4 shadow-gov-sm space-y-3">
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider border-b pb-2 flex items-center gap-1.5">
            <BellRing className="w-4 h-4 text-slate-700" />
            Automated Warning Dissemination Channels
          </h3>

          <div className="space-y-2 text-xs">
            <label className="flex items-center justify-between p-2.5 rounded bg-slate-50 border border-slate-200 cursor-pointer">
              <div>
                <span className="font-bold text-slate-800 block">NDMA Common Alerting Protocol (CAP) Integration</span>
                <span className="text-[11px] text-slate-500">Automatically broadcast localized XML alerts to cellular base stations</span>
              </div>
              <input
                type="checkbox"
                checked={capAlerts}
                onChange={(e) => setCapAlerts(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded"
              />
            </label>

            <div className="flex items-center justify-between p-2.5 rounded bg-slate-50 border border-slate-200">
              <div>
                <span className="font-bold text-slate-800 block">Telemetry Polling Frequency</span>
                <span className="text-[11px] text-slate-500">INSAT-3DR Rapid Scan and DWR radar sync frequency</span>
              </div>
              <select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                className="bg-white border border-slate-300 rounded px-2.5 py-1 text-xs font-semibold"
              >
                <option value={5}>Every 5 Minutes (Rapid Scan)</option>
                <option value={10}>Every 10 Minutes</option>
                <option value={15}>Every 15 Minutes (Standard)</option>
                <option value={30}>Every 30 Minutes</option>
              </select>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="flex items-center justify-between pt-2">
          {saved && (
            <span className="text-xs font-bold text-emerald-700 flex items-center gap-1.5 bg-emerald-50 px-3 py-1.5 rounded border border-emerald-200">
              <ShieldCheck className="w-4 h-4" />
              Settings successfully saved to local station cache!
            </span>
          )}
          <button
            type="submit"
            className="ml-auto px-4 py-2 bg-[#0A3871] hover:bg-[#07264F] text-white text-xs font-bold rounded flex items-center gap-2 shadow-sm transition-colors"
          >
            <Save className="w-4 h-4" />
            Save Configuration
          </button>
        </div>
      </form>
    </div>
  );
};

export default SettingsPage;
