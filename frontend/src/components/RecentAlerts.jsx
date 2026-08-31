import React from 'react';
import { AlertTriangle, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useAppStore } from '../store/useAppStore';

export const RecentAlerts = () => {
  const { alerts, setSelectedAlert } = useAppStore();

  return (
    <div className="bg-white border border-[#E2E2E2] rounded p-3 shadow-gov-sm flex flex-col justify-between h-full">
      {/* Header */}
      <div className="border-b border-[#E2E2E2] pb-1.5 mb-2">
        <h3 className="text-xs sm:text-[13px] font-bold text-[#172033] m-0 flex items-center gap-1.5">
          <span className="w-1 h-3.5 bg-[#E87516] rounded-sm inline-block"></span>
          <span>Recent Alerts</span>
        </h3>
      </div>

      {/* Alert Items List */}
      <div className="space-y-1.5">
        {alerts.slice(0, 3).map((item) => (
          <div
            key={item.id}
            onClick={() => setSelectedAlert(item)}
            className="flex items-start justify-between gap-2 p-1 rounded hover:bg-[#FFF7EA] transition-colors cursor-pointer"
            title="Click to view full bulletin details"
          >
            <div className="flex items-start gap-1.5 min-w-0">
              <AlertTriangle
                className={`w-3.5 h-3.5 mt-0.5 shrink-0 ${
                  item.severity === 'danger'
                    ? 'text-red-600'
                    : item.severity === 'warning'
                    ? 'text-orange-500'
                    : 'text-amber-500'
                }`}
              />
              <div className="min-w-0">
                <span className="text-[11.5px] font-bold text-[#172033] block truncate leading-tight">
                  {item.type}
                </span>
                <span className="text-[10px] text-slate-500 block truncate leading-tight">
                  {item.location}
                </span>
              </div>
            </div>
            <span className="text-[10px] text-slate-500 font-semibold whitespace-nowrap pt-0.5">
              {item.time}
            </span>
          </div>
        ))}
      </div>

      {/* Footer Link */}
      <div className="pt-1.5 border-t border-slate-100 mt-2">
        <Link
          to="/alerts"
          className="inline-flex items-center gap-1 text-[11px] font-bold text-[#C85D00] hover:text-[#A84A00] hover:underline"
        >
          <span>View All Alerts</span>
          <ArrowRight className="w-3 h-3" />
        </Link>
      </div>
    </div>
  );
};

export default RecentAlerts;
