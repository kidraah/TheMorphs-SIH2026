import React, { useState } from 'react';
import { FileText, Download, Printer, CheckCircle, Calendar } from 'lucide-react';

export const ReportsPage = () => {
  const [selectedBulletin, setSelectedBulletin] = useState('nowcast-01');

  const bulletins = [
    {
      id: 'nowcast-01',
      title: 'SPECIAL NOWCAST BULLETIN #IMD-NW-2026/0829-04',
      date: '29 August 2026, 10:30 IST',
      category: 'Cloudburst & Severe Weather Warning',
      signatory: 'Dr. S. K. Roy, Head - Hydromet Division, IMD New Delhi',
    },
    {
      id: 'nowcast-02',
      title: 'FLASH FLOOD ADVISORY BULLETIN #IMD-FFA-2026/0829-02',
      date: '29 August 2026, 09:00 IST',
      category: 'Hydrological Inundation Advisory',
      signatory: 'Director, Regional Meteorological Centre (RMC), New Delhi',
    },
    {
      id: 'nowcast-03',
      title: 'DAILY SEVERE WEATHER SUMMARY & VERIFICATION REPORT',
      date: '28 August 2026, 18:00 IST',
      category: 'Daily Performance Review',
      signatory: 'AI Nowcasting Operations Center, MoES',
    },
  ];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-white border border-[#E2E2E2] rounded p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-gov-sm">
        <div>
          <h2 className="text-base font-bold text-[#172033] m-0 flex items-center gap-2">
            <span className="w-1.5 h-4.5 bg-[#E87516] rounded-sm inline-block"></span>
            <FileText className="w-5 h-5 text-[#E87516]" />
            Official Meteorological Bulletins & Disaster SOP Reports
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 m-0">
            Standard Operating Procedure (SOP) compliant reports formatted for NDMA, SDMA, and District Collectors
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Bulletins List (Left) */}
        <div className="lg:col-span-4 space-y-2.5">
          <h3 className="text-xs font-bold text-[#172033] uppercase tracking-wide">
            Recent Issued Bulletins
          </h3>
          {bulletins.map((b) => (
            <div
              key={b.id}
              onClick={() => setSelectedBulletin(b.id)}
              className={`p-3 rounded border cursor-pointer transition-all ${
                selectedBulletin === b.id
                  ? 'bg-[#FFF1DD] border-[#E87516] shadow-sm'
                  : 'bg-white border-[#E2E2E2] hover:bg-[#FFF7EA]'
              }`}
            >
              <div className="text-[11px] font-bold text-red-700 mb-1">{b.category}</div>
              <div className="text-xs font-bold text-[#172033] leading-snug">{b.title}</div>
              <div className="text-[10.5px] text-slate-500 mt-1 flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                {b.date}
              </div>
            </div>
          ))}
        </div>

        {/* Bulletin Viewer (Right) */}
        <div className="lg:col-span-8 bg-white border border-[#E2E2E2] rounded p-6 shadow-gov-sm font-sans space-y-4">
          <div className="flex items-center justify-between border-b border-[#E2E2E2] pb-3">
            <div>
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block">
                GOVERNMENT OF INDIA • MINISTRY OF EARTH SCIENCES
              </span>
              <h2 className="text-sm font-extrabold text-[#E87516] mt-0.5">
                INDIA METEOROLOGICAL DEPARTMENT
              </h2>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => window.print()}
                className="px-2.5 py-1 text-xs border border-slate-300 rounded font-semibold text-slate-700 hover:bg-[#FFF7EA] flex items-center gap-1"
              >
                <Printer className="w-3.5 h-3.5" /> Print
              </button>
              <button
                onClick={() => window.print()}
                className="px-2.5 py-1 text-xs bg-[#E87516] text-white rounded font-semibold hover:bg-[#C85D00] flex items-center gap-1"
              >
                <Download className="w-3.5 h-3.5" /> PDF
              </button>
            </div>
          </div>

          <div className="space-y-3 text-xs text-slate-800 leading-relaxed border border-[#E2E2E2] p-4 bg-slate-50/50 rounded">
            <p className="font-bold text-red-700">
              URGENT / IMMEDIATE ATTENTION: STATE EMERGENCY OPERATIONS CENTER (SEOC), DEHRADUN & SHIMLA
            </p>
            <p>
              <strong>1. Synopsis:</strong> Multi-sensor convective analysis from INSAT-3DR and Doppler Weather Radar indicates severe convective cloud cluster convergence over Chamoli, Rudraprayag, and Tehri Garhwal districts of Uttarakhand.
            </p>
            <p>
              <strong>2. Threat Evaluation:</strong>
              <br />• <strong>Cloudburst Likelihood:</strong> 85% probability in Mandakini & Alaknanda upper catchments between 11:00 IST and 15:30 IST.
              <br />• <strong>Flash Flood Threat:</strong> High inundation probability in low-lying settlements along riverbanks.
              <br />• <strong>Thunderstorm & Lightning:</strong> Intense cloud-to-ground strikes expected across 12 Himalayan districts.
            </p>
            <p>
              <strong>3. Action Recommended:</strong>
              <br />• Immediate suspension of pilgrim and tourist movement along sensitive river corridors.
              <br />• Pre-positioning of NDRF and SDRF rescue battalions in Joshimath, Karnaprayag, and Ukhimath.
              <br />• Real-time monitoring of automated river gauge telemetry.
            </p>
          </div>

          <div className="pt-2 flex justify-between items-end text-xs text-slate-600 border-t border-[#E2E2E2]">
            <div>
              <span className="block font-bold text-[#172033]">Issued by:</span>
              <span>National Weather Forecasting Centre (NWFC), New Delhi</span>
            </div>
            <div className="text-right">
              <span className="block font-bold text-emerald-800 flex items-center gap-1 justify-end">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-600" /> Digitally Authenticated
              </span>
              <span className="text-[10px] text-slate-500">IMD AI Warning System Core v2.4</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportsPage;
