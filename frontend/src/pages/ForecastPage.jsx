import React from 'react';
import { CloudSun, Wind, Thermometer, Droplets, Activity } from 'lucide-react';

export const ForecastPage = () => {
  const thermodynamicIndices = [
    { name: 'Convective Available Potential Energy (CAPE)', value: '3,850 J/kg', status: 'Extremely Unstable', color: 'text-red-600' },
    { name: 'Convective Inhibition (CIN)', value: '-12 J/kg', status: 'Cap Broken', color: 'text-red-600' },
    { name: 'Precipitable Water (PW)', value: '62 mm', status: 'Saturated Column', color: 'text-purple-700' },
    { name: 'Lifted Index (LI)', value: '-7.5 °C', status: 'Severe Convective Potential', color: 'text-red-600' },
    { name: 'K-Index', value: '39 °C', status: 'High Thunderstorm Likelihood', color: 'text-orange-600' },
    { name: 'Total Totals Index (TT)', value: '54 °C', status: 'Severe Storms Probable', color: 'text-orange-600' },
  ];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-white border border-[#E2E2E2] rounded p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-gov-sm">
        <div>
          <h2 className="text-base font-bold text-[#172033] m-0 flex items-center gap-2">
            <span className="w-1.5 h-4.5 bg-[#E87516] rounded-sm inline-block"></span>
            <CloudSun className="w-5 h-5 text-[#E87516]" />
            Atmospheric Diagnostics & Convective Nowcast Profiles
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 m-0">
            High-resolution NWP model assimilation & Doppler Weather Radar vertical soundings
          </p>
        </div>
      </div>

      {/* Thermodynamic Indices Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {thermodynamicIndices.map((item, idx) => (
          <div key={idx} className="bg-white border border-slate-200 rounded p-3 shadow-gov-sm">
            <div className="text-[11px] font-bold text-slate-600 uppercase tracking-wide mb-1">
              {item.name}
            </div>
            <div className="flex items-baseline justify-between mt-2">
              <span className={`text-xl font-extrabold ${item.color}`}>{item.value}</span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200">
                {item.status}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Synoptic Sounding Chart Card */}
      <div className="bg-white border border-slate-200 rounded p-3.5 shadow-gov-sm">
        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2.5 border-b pb-1.5 flex items-center gap-1.5">
          <Activity className="w-4 h-4 text-blue-600" />
          Upper-Air Sounding & Moisture Flux Divergence (00Z/12Z Assimilation)
        </h3>
        <div className="h-56 bg-slate-50 border border-slate-200 rounded flex items-center justify-center p-4">
          <div className="w-full text-center space-y-2">
            <div className="flex justify-around items-end h-32 px-8 border-b border-slate-300">
              <div className="w-12 bg-blue-300 rounded-t h-16 flex items-center justify-center text-[10px] font-bold">850 hPa</div>
              <div className="w-12 bg-blue-400 rounded-t h-24 flex items-center justify-center text-[10px] font-bold text-white">700 hPa</div>
              <div className="w-12 bg-blue-500 rounded-t h-28 flex items-center justify-center text-[10px] font-bold text-white">500 hPa</div>
              <div className="w-12 bg-purple-600 rounded-t h-32 flex items-center justify-center text-[10px] font-bold text-white">300 hPa</div>
              <div className="w-12 bg-red-600 rounded-t h-20 flex items-center justify-center text-[10px] font-bold text-white">200 hPa</div>
            </div>
            <p className="text-xs text-slate-600 font-medium">
              Significant moisture convergence layer between 850 hPa - 500 hPa with intense orographic lifting along Himalayan foothills.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForecastPage;
