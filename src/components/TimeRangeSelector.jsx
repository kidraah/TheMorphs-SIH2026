import React from 'react';
import { Clock } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

export const TimeRangeSelector = () => {
  const { selectedTimeRange, setSelectedTimeRange } = useAppStore();

  const timeOptions = [
    { id: 'now', label: 'Now' },
    { id: '2h', label: 'Next 2 Hours' },
    { id: '4h', label: 'Next 4 Hours' },
    { id: '6h', label: 'Next 6 Hours' },
  ];

  return (
    <div className="bg-slate-50 border border-slate-200 rounded p-2.5">
      <div className="flex items-center gap-1.5 mb-1.5">
        <Clock className="w-3.5 h-3.5 text-slate-500" />
        <label className="text-[11px] font-bold text-slate-700 uppercase tracking-wider">
          Time Range
        </label>
      </div>
      <div className="grid grid-cols-2 gap-1.5 bg-slate-200/70 p-1 rounded">
        {timeOptions.map((opt) => (
          <button
            key={opt.id}
            type="button"
            onClick={() => setSelectedTimeRange(opt.id)}
            className={`py-1 text-[11px] font-semibold rounded transition-all text-center ${
              selectedTimeRange === opt.id
                ? 'bg-[#E87516] text-white shadow-sm font-bold'
                : 'bg-white/80 text-slate-700 hover:bg-white hover:text-slate-900 border border-slate-200/60'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
};

export default TimeRangeSelector;
