import React from 'react';
import { ChevronDown, MapPin } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

export const RegionSelector = () => {
  const { selectedRegion, setSelectedRegion, regions } = useAppStore();

  return (
    <div className="bg-slate-50 border border-slate-200 rounded p-2.5">
      <div className="flex items-center gap-1.5 mb-1.5">
        <MapPin className="w-3.5 h-3.5 text-slate-500" />
        <label className="text-[11px] font-bold text-slate-700 uppercase tracking-wider">
          Region Selector
        </label>
      </div>
      <div className="relative">
        <select
          value={selectedRegion}
          onChange={(e) => setSelectedRegion(e.target.value)}
          className="w-full bg-white text-xs font-medium text-slate-800 border border-slate-300 rounded py-1.5 pl-2.5 pr-7 appearance-none cursor-pointer focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-sm"
        >
          {regions.map((r) => (
            <option key={r.id} value={r.id}>
              {r.name}
            </option>
          ))}
        </select>
        <ChevronDown className="w-3.5 h-3.5 text-slate-500 absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none" />
      </div>
    </div>
  );
};

export default RegionSelector;
