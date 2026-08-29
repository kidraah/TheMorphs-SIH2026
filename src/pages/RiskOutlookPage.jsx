import React, { useState } from 'react';
import { DISTRICT_RISK_DATA } from '../data/districtData';
import { TrendingUp, AlertTriangle, ShieldCheck, Download, Filter } from 'lucide-react';

export const RiskOutlookPage = () => {
  const [filterState, setFilterState] = useState('all');
  const [filterRisk, setFilterRisk] = useState('all');

  const filteredDistricts = DISTRICT_RISK_DATA.filter((d) => {
    if (filterState !== 'all' && d.state !== filterState) return false;
    if (filterRisk !== 'all' && d.riskLevel.toLowerCase() !== filterRisk.toLowerCase()) return false;
    return true;
  });

  const getBadgeStyle = (level) => {
    switch (level.toLowerCase()) {
      case 'extreme':
        return 'bg-purple-100 text-purple-800 border-purple-300';
      case 'very high':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'high':
        return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'moderate':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      default:
        return 'bg-emerald-100 text-emerald-800 border-emerald-300';
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-white border border-[#E2E2E2] rounded p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-gov-sm">
        <div>
          <h2 className="text-base font-bold text-[#172033] m-0 flex items-center gap-2">
            <span className="w-1.5 h-4.5 bg-[#E87516] rounded-sm inline-block"></span>
            <TrendingUp className="w-5 h-5 text-[#E87516]" />
            Comprehensive District Risk Outlook & Vulnerability Matrix
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 m-0">
            Synoptic 24-hour and 48-hour convective risk forecasts for state and district disaster managers
          </p>
        </div>
        <button
          type="button"
          onClick={() => window.print()}
          className="px-3 py-1.5 bg-[#E87516] hover:bg-[#C85D00] text-white text-xs font-semibold rounded flex items-center gap-1.5 shadow-sm transition-colors"
        >
          <Download className="w-3.5 h-3.5" />
          Export Risk Bulletin (PDF)
        </button>
      </div>

      {/* Filter Bar */}
      <div className="bg-white border border-[#E2E2E2] rounded p-3 shadow-gov-sm flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-slate-500" />
            <span className="font-bold text-[#172033]">Filter By State:</span>
            <select
              value={filterState}
              onChange={(e) => setFilterState(e.target.value)}
              className="bg-slate-50 border border-slate-300 rounded px-2 py-1 text-xs"
            >
              <option value="all">All States</option>
              <option value="Uttarakhand">Uttarakhand</option>
              <option value="Himachal Pradesh">Himachal Pradesh</option>
            </select>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="font-bold text-[#172033]">Risk Severity:</span>
            <select
              value={filterRisk}
              onChange={(e) => setFilterRisk(e.target.value)}
              className="bg-slate-50 border border-slate-300 rounded px-2 py-1 text-xs"
            >
              <option value="all">All Severity Levels</option>
              <option value="extreme">Extreme</option>
              <option value="very high">Very High</option>
              <option value="high">High</option>
              <option value="moderate">Moderate</option>
              <option value="low">Low</option>
            </select>
          </div>
        </div>

        <span className="text-slate-500 font-medium">
          Showing <strong>{filteredDistricts.length}</strong> of {DISTRICT_RISK_DATA.length} Districts
        </span>
      </div>

      {/* District Data Table */}
      <div className="bg-white border border-[#E2E2E2] rounded shadow-gov-sm overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="bg-slate-100/90 text-slate-700 border-b border-slate-300 font-bold uppercase text-[10.5px]">
              <th className="p-2.5">District</th>
              <th className="p-2.5">State</th>
              <th className="p-2.5">Risk Level</th>
              <th className="p-2.5">Cloudburst Prob</th>
              <th className="p-2.5">Flash Flood Prob</th>
              <th className="p-2.5">Nowcast Precip</th>
              <th className="p-2.5">Population Exposed</th>
              <th className="p-2.5">River Catchment</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200">
            {filteredDistricts.map((item) => (
              <tr key={item.id} className="hover:bg-[#FFF7EA]/40 transition-colors">
                <td className="p-2.5 font-bold text-[#172033]">{item.name}</td>
                <td className="p-2.5 text-slate-600">{item.state}</td>
                <td className="p-2.5">
                  <span className={`px-2 py-0.5 rounded border font-bold text-[10px] ${getBadgeStyle(item.riskLevel)}`}>
                    {item.riskLevel}
                  </span>
                </td>
                <td className="p-2.5 font-semibold text-purple-700">{item.cloudburstProb}%</td>
                <td className="p-2.5 font-semibold text-blue-700">{item.flashFloodProb}%</td>
                <td className="p-2.5 font-bold text-slate-800">{item.rainfall24h}</td>
                <td className="p-2.5 text-slate-700">{item.populationExposed}</td>
                <td className="p-2.5 text-slate-600 font-medium">{item.rivers}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default RiskOutlookPage;
