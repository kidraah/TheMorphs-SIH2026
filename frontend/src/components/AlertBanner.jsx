import React from 'react';
import { AlertTriangle, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useAppStore } from '../store/useAppStore';

export const AlertBanner = () => {
  const { alerts, acknowledgedAlerts, setSelectedAlert } = useAppStore();

  // Find the first active unacknowledged alert
  const activeAlert = alerts.find((a) => !acknowledgedAlerts.includes(a.id)) || alerts[0];

  if (!activeAlert) {
    return (
      <div
        className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 px-3 py-2 rounded border border-emerald-200 bg-emerald-50 text-emerald-800"
      >
        <div className="flex items-center gap-2.5">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="text-[12px] font-bold">No active warnings. Weather situation is stable.</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 px-3 py-2 rounded"
      style={{
        background: activeAlert.severity === 'danger' ? '#FEF2F2' : '#FFF7ED',
        border: activeAlert.severity === 'danger' ? '1px solid #FECACA' : '1px solid #FFEDD5',
      }}
    >
      {/* Warning Icon + Text (Clickable to inspect) */}
      <div
        onClick={() => setSelectedAlert(activeAlert)}
        className="flex items-center gap-2.5 min-w-0 cursor-pointer group flex-1"
        title="Click to view detailed warning bulletin"
      >
        <div className={`rounded p-1 shrink-0 ${
          activeAlert.severity === 'danger' ? 'bg-red-100' : 'bg-orange-100'
        }`}>
          <AlertTriangle className={`w-4 h-4 ${
            activeAlert.severity === 'danger' ? 'text-red-600' : 'text-orange-500'
          }`} />
        </div>
        <p className="text-[12px] text-slate-800 leading-snug m-0 group-hover:text-[#C85D00] transition-colors">
          <strong className={`${
            activeAlert.severity === 'danger' ? 'text-red-700' : 'text-orange-700'
          } font-bold mr-1`}>
            {activeAlert.type} [{activeAlert.location}]:
          </strong>
          {activeAlert.details}
        </p>
      </div>

      {/* Action Link */}
      <Link
        to="/alerts"
        className="inline-flex items-center gap-1 text-[12px] font-bold text-[#C85D00] hover:underline shrink-0 self-end sm:self-center whitespace-nowrap ml-4"
      >
        View All Alerts
        <ArrowRight className="w-3.5 h-3.5" />
      </Link>
    </div>
  );
};

export default AlertBanner;
