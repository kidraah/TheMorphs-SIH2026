import React from 'react';
import { Droplet, CloudSnow, Zap, Wind, Mountain, Compass } from 'lucide-react';
import { mockXaiTriggers } from '../data/mockData';

const getXaiIcon = (iconType) => {
  switch (iconType) {
    case 'droplet':
      return <Droplet className="w-3.5 h-3.5 text-sky-600 shrink-0" />;
    case 'cloud-snow':
      return <CloudSnow className="w-3.5 h-3.5 text-blue-500 shrink-0" />;
    case 'zap':
      return <Zap className="w-3.5 h-3.5 text-amber-500 shrink-0 fill-current" />;
    case 'wind':
      return <Wind className="w-3.5 h-3.5 text-teal-600 shrink-0" />;
    case 'compass':
      return <Compass className="w-3.5 h-3.5 text-indigo-600 shrink-0" />;
    case 'mountain':
      return <Mountain className="w-3.5 h-3.5 text-emerald-600 shrink-0" />;
    default:
      return <Zap className="w-3.5 h-3.5 text-slate-600 shrink-0" />;
  }
};

export const ExplainableAI = ({ triggers = mockXaiTriggers }) => {
  return (
    <div className="bg-white border border-[#E2E2E2] rounded p-3 shadow-gov-sm flex flex-col justify-between h-full">
      {/* Header */}
      <div className="border-b border-[#E2E2E2] pb-1.5 mb-1.5 flex items-center justify-between">
        <h3 className="text-xs sm:text-[13px] font-bold text-[#172033] m-0 flex items-center gap-1.5">
          <span className="w-1 h-3.5 bg-[#E87516] rounded-sm inline-block"></span>
          <span>Explainable AI – Key Triggers</span>
        </h3>
      </div>

      {/* Trigger Items List */}
      <div className="divide-y divide-slate-100">
        {triggers.map((item) => (
          <div key={item.id} className="py-1 first:pt-0 last:pb-0 flex items-center justify-between gap-2">
            <div className="flex items-start gap-2 min-w-0">
              <div className="mt-0.5">{getXaiIcon(item.iconType)}</div>
              <div className="min-w-0">
                <div className="text-[11.5px] font-bold text-slate-900 truncate leading-tight">
                  {item.title}
                </div>
                <div className="text-[10px] text-slate-500 truncate leading-tight">
                  {item.description}
                </div>
              </div>
            </div>
            <div className="shrink-0 text-right">
              <span className="text-[11.5px] font-bold text-red-600">
                {item.impact}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExplainableAI;
