import React, { useState } from 'react';
import { Bell, AlertTriangle, CheckCircle, ShieldAlert, Clock, Check } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

export const AlertsPage = () => {
  const { alerts, acknowledgedAlerts, acknowledgeAlert } = useAppStore();
  const [filterSeverity, setFilterSeverity] = useState('all');

  const filtered = alerts.filter((a) => {
    if (filterSeverity === 'acknowledged') return acknowledgedAlerts.includes(a.id);
    if (filterSeverity === 'active') return !acknowledgedAlerts.includes(a.id);
    return true;
  });

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-white border border-[#E2E2E2] rounded p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-gov-sm">
        <div>
          <h2 className="text-base font-bold text-[#172033] m-0 flex items-center gap-2">
            <span className="w-1.5 h-4.5 bg-[#E87516] rounded-sm inline-block"></span>
            <Bell className="w-5 h-5 text-red-600" />
            Active Warning Bulletins & Nowcast Alerts
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 m-0">
            Real-time critical severe weather alerts issued to State Emergency Operations Centers (SEOC)
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="px-2.5 py-1 bg-red-100 text-red-800 font-bold rounded border border-red-200">
            {alerts.length - acknowledgedAlerts.length} Actionable Warnings
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-2 border-b border-[#E2E2E2] bg-white px-3 pt-2 rounded-t text-xs">
        <button
          onClick={() => setFilterSeverity('all')}
          className={`pb-2 px-2 font-bold border-b-2 transition-all ${
            filterSeverity === 'all'
              ? 'border-[#E87516] text-[#C85D00]'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          All Bulletins ({alerts.length})
        </button>
        <button
          onClick={() => setFilterSeverity('active')}
          className={`pb-2 px-2 font-bold border-b-2 transition-all ${
            filterSeverity === 'active'
              ? 'border-red-600 text-red-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          Active / Unacknowledged ({alerts.length - acknowledgedAlerts.length})
        </button>
        <button
          onClick={() => setFilterSeverity('acknowledged')}
          className={`pb-2 px-2 font-bold border-b-2 transition-all ${
            filterSeverity === 'acknowledged'
              ? 'border-emerald-600 text-emerald-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          Acknowledged Logs ({acknowledgedAlerts.length})
        </button>
      </div>

      {/* Alerts Feed */}
      <div className="space-y-3">
        {filtered.map((item) => {
          const isAck = acknowledgedAlerts.includes(item.id);
          return (
            <div
              key={item.id}
              className={`bg-white border rounded p-3.5 shadow-gov-sm transition-all ${
                isAck
                  ? 'border-slate-200 opacity-75'
                  : item.severity === 'danger'
                  ? 'border-red-300 bg-red-50/30'
                  : 'border-orange-300 bg-orange-50/20'
              }`}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-2 mb-2">
                <div className="flex items-center gap-2">
                  <AlertTriangle
                    className={`w-4 h-4 shrink-0 ${
                      item.severity === 'danger' ? 'text-red-600' : 'text-orange-500'
                    }`}
                  />
                  <h3 className="text-sm font-bold text-slate-900 m-0">{item.type}</h3>
                  <span className="text-xs px-2 py-0.5 bg-slate-100 font-semibold text-slate-700 rounded border border-slate-200">
                    {item.location}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs text-slate-500">
                  <span className="flex items-center gap-1 font-mono">
                    <Clock className="w-3.5 h-3.5 text-slate-400" />
                    Issued: {item.time}
                  </span>
                  {isAck && (
                    <span className="flex items-center gap-1 text-emerald-700 font-bold bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                      <Check className="w-3 h-3" /> Acknowledged
                    </span>
                  )}
                </div>
              </div>

              <p className="text-xs text-slate-700 leading-relaxed mb-3">
                {item.details}
              </p>

              <div className="flex items-center justify-between pt-1">
                <div className="text-[11px] text-slate-500 font-medium">
                  Source: <strong>IMD Nowcast AI Engine v2.4</strong> | Lead Time: <strong>2-6 Hours</strong>
                </div>
                {!isAck && (
                  <button
                    type="button"
                    onClick={() => acknowledgeAlert(item.id)}
                    className="px-2.5 py-1 bg-white hover:bg-slate-50 text-slate-700 font-semibold text-xs border border-slate-300 rounded shadow-sm flex items-center gap-1.5 transition-colors"
                  >
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                    Acknowledge Alert
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default AlertsPage;
