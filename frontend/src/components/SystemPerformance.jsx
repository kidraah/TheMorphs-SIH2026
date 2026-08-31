import React from 'react';
import { ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

export const SystemPerformance = ({ metrics = [] }) => {
  return (
    <div className="bg-white border border-[#E2E2E2] rounded p-3 shadow-gov-sm flex flex-col justify-between h-full">
      {/* Header */}
      <div className="border-b border-[#E2E2E2] pb-1.5 mb-2">
        <h3 className="text-xs sm:text-[13px] font-bold text-[#172033] m-0 flex items-center gap-1.5">
          <span className="w-1 h-3.5 bg-[#E87516] rounded-sm inline-block"></span>
          <span>System Performance</span>
        </h3>
      </div>

      {/* Metric Progress Bars */}
      <div className="space-y-2">
        {metrics.map((item, idx) => (
          <div key={idx} className="space-y-1">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-slate-700 font-medium">{item.name}</span>
              <span className="font-bold text-[#172033]">{item.percentage}%</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden border border-[#E2E2E2]">
              <div
                className="bg-[#E87516] h-full rounded-full transition-all duration-500"
                style={{ width: `${item.percentage}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      {/* Footer Link */}
      <div className="pt-1.5 border-t border-slate-100 mt-2">
        <Link
          to="/about-system"
          className="inline-flex items-center gap-1 text-[11px] font-bold text-[#C85D00] hover:text-[#A84A00] hover:underline"
        >
          <span>View Performance Details</span>
          <ArrowRight className="w-3 h-3" />
        </Link>
      </div>
    </div>
  );
};

export default SystemPerformance;
