import React from 'react';
import { X, AlertTriangle, Clock, MapPin, CheckCircle, ShieldAlert, Users, Navigation } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

export const AlertDetailModal = () => {
  const { selectedAlert, clearSelectedAlert, acknowledgeAlert, acknowledgedAlerts } = useAppStore();

  if (!selectedAlert) return null;

  const isAck = acknowledgedAlerts.includes(selectedAlert.id);

  return (
    <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
      {/* Modal Card */}
      <div className="bg-white border border-slate-300 rounded-lg shadow-2xl max-w-lg w-full overflow-hidden flex flex-col font-sans">
        
        {/* Modal Header */}
        <div className={`p-4 text-white flex items-center justify-between ${
          selectedAlert.severity === 'danger' ? 'bg-[#DC2626]' : 'bg-[#EA580C]'
        }`}>
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 animate-pulse" />
            <div>
              <h3 className="font-extrabold text-sm uppercase tracking-wider leading-none">
                IMD Severe Weather Bulletin
              </h3>
              <span className="text-[10px] opacity-90 mt-0.5 block font-mono">
                ID: {selectedAlert.id} | Level: {selectedAlert.severity?.toUpperCase()}
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={clearSelectedAlert}
            className="text-white/80 hover:text-white hover:bg-white/10 p-1 rounded-full transition-colors"
          >
            <X className="w-4.5 h-4.5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-4 space-y-4 overflow-y-auto max-h-[70vh] text-xs">
          {/* Main Summary */}
          <div className="bg-slate-50 border border-slate-200 rounded p-3">
            <p className="text-[13px] font-extrabold text-[#C85D00]">
              {selectedAlert.type}
            </p>
            <p className="text-slate-500 font-semibold mt-0.5 flex items-center gap-1">
              <MapPin className="w-3.5 h-3.5 text-red-600" />
              {selectedAlert.location}
            </p>
            <p className="text-slate-700 mt-2 leading-relaxed font-medium">
              {selectedAlert.details}
            </p>
          </div>

          {/* Details Table Grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-50 border border-slate-200 rounded p-2.5">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">
                Issued By
              </span>
              <span className="font-bold text-slate-800">
                {selectedAlert.issuedBy || 'IMD Central Division'}
              </span>
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded p-2.5">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">
                Valid Until
              </span>
              <span className="font-bold text-red-600 flex items-center gap-1">
                <Clock className="w-3.5 h-3.5" />
                {selectedAlert.validUntil || 'Next 2 Hours'}
              </span>
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded p-2.5">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">
                Rainfall Range
              </span>
              <span className="font-extrabold text-blue-700">
                {selectedAlert.expectedRainfall || '50–100 mm'}
              </span>
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded p-2.5">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">
                Wind Conditions
              </span>
              <span className="font-bold text-slate-800">
                {selectedAlert.windSpeed || '40–60 km/h'}
              </span>
            </div>
          </div>

          {/* Exposure and Rivers info */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-50 border border-slate-200 rounded p-2.5 flex items-center gap-2">
              <Users className="w-5 h-5 text-emerald-600 shrink-0" />
              <div>
                <span className="text-[9px] text-slate-400 font-bold uppercase block">
                  Exposed Population
                </span>
                <span className="font-bold text-slate-800">
                  {selectedAlert.affectedPopulation || 'Moderate'}
                </span>
              </div>
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded p-2.5 flex items-center gap-2">
              <Navigation className="w-5 h-5 text-sky-600 shrink-0" />
              <div>
                <span className="text-[9px] text-slate-400 font-bold uppercase block">
                  Key Catchments
                </span>
                <span className="font-bold text-slate-800 truncate block max-w-[150px]">
                  {selectedAlert.rivers || 'Localized Drainage'}
                </span>
              </div>
            </div>
          </div>

          {/* Actionable Instructions */}
          <div className="bg-amber-50 border border-amber-200 rounded p-3 text-amber-900">
            <span className="font-bold text-[11px] block mb-1 uppercase tracking-wide flex items-center gap-1">
              <ShieldAlert className="w-3.5 h-3.5 text-amber-700" />
              Actionable Directives (SDMA/SEOC):
            </span>
            <p className="font-medium leading-relaxed">
              {selectedAlert.recommendedAction || 'Monitor local streams and take shelters on high terrain.'}
            </p>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-3 border-t border-slate-200 bg-slate-50 flex items-center justify-between gap-3">
          <div className="text-[10px] text-slate-400 font-medium">
            Transmission Protocol: Satellite (GSAT-29) / Cell-Broadcast
          </div>
          <div className="flex gap-2">
            {!isAck && (
              <button
                type="button"
                onClick={() => {
                  acknowledgeAlert(selectedAlert.id);
                  clearSelectedAlert();
                }}
                className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-xs rounded shadow flex items-center gap-1.5 transition-colors"
              >
                <CheckCircle className="w-3.5 h-3.5" />
                Acknowledge Bulletin
              </button>
            )}
            <button
              type="button"
              onClick={clearSelectedAlert}
              className="px-3.5 py-1.5 bg-white hover:bg-slate-100 text-slate-700 font-semibold text-xs border border-slate-300 rounded shadow-sm transition-colors"
            >
              Close
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};

export default AlertDetailModal;
