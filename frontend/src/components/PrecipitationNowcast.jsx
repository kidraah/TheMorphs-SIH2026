import React from 'react';

export const PrecipitationNowcast = () => {
  return (
    <div className="bg-white border border-[#E2E2E2] rounded p-3 shadow-gov-sm flex flex-col justify-between h-full">
      {/* Header */}
      <div className="border-b border-[#E2E2E2] pb-1.5 mb-2">
        <h3 className="text-xs sm:text-[13px] font-bold text-[#172033] m-0 flex items-center gap-1.5">
          <span className="w-1 h-3.5 bg-[#E87516] rounded-sm inline-block"></span>
          <span>Precipitation Nowcast</span>
          <span className="text-slate-500 font-normal text-xs">(mm)</span>
        </h3>
      </div>

      {/* Radar Map with Vertical Scale */}
      <div className="flex items-center gap-2.5 h-32 my-auto">
        {/* Radar Heatmap Canvas/SVG */}
        <div className="relative flex-1 h-full bg-[#0F284E] rounded border border-slate-300 overflow-hidden">
          <svg viewBox="0 0 200 120" className="w-full h-full object-cover">
            {/* Topographic river / valley contour lines */}
            <path d="M0,45 Q60,25 110,55 T200,35" fill="none" stroke="#1E4B7A" strokeWidth="0.7" opacity="0.6" />
            <path d="M0,85 Q80,95 140,65 T200,95" fill="none" stroke="#1E4B7A" strokeWidth="0.7" opacity="0.6" />
            <path d="M45,0 Q65,65 55,120" fill="none" stroke="#1E4B7A" strokeWidth="0.7" opacity="0.6" />
            <path d="M155,0 Q135,55 165,120" fill="none" stroke="#1E4B7A" strokeWidth="0.7" opacity="0.6" />

            {/* Radar Convective Precipitation Echo Zones */}
            {/* 10-30 mm (Cyan/Blue) */}
            <ellipse cx="115" cy="62" rx="72" ry="42" fill="#0284C7" opacity="0.45" />
            <ellipse cx="75" cy="52" rx="42" ry="30" fill="#06B6D4" opacity="0.45" />
            
            {/* 30-60 mm (Green) */}
            <ellipse cx="118" cy="62" rx="52" ry="30" fill="#16A34A" opacity="0.65" />
            <ellipse cx="78" cy="52" rx="28" ry="18" fill="#16A34A" opacity="0.55" />

            {/* 60-90 mm (Yellow) */}
            <ellipse cx="120" cy="64" rx="34" ry="20" fill="#EAB308" opacity="0.80" />

            {/* 90-120 mm (Orange) */}
            <ellipse cx="122" cy="66" rx="22" ry="13" fill="#EA580C" opacity="0.90" />

            {/* 120-150 mm (Red Core) */}
            <ellipse cx="124" cy="67" rx="13" ry="8" fill="#DC2626" opacity="0.95" />

            {/* 150+ mm (Purple Extreme Hotspot) */}
            <ellipse cx="125" cy="67" rx="6" ry="4" fill="#7E22CE" opacity="0.98" />
          </svg>
        </div>

        {/* Vertical Colorbar Scale */}
        <div className="flex items-center gap-1.5 h-full shrink-0">
          <div className="w-2.5 h-full rounded-sm bg-gradient-to-t from-[#0A3871] via-[#06B6D4] via-[#16A34A] via-[#EAB308] via-[#EA580C] via-[#DC2626] to-[#7E22CE] border border-slate-300"></div>
          <div className="flex flex-col justify-between h-full text-[9px] font-bold text-slate-600 select-none leading-none">
            <span>150+</span>
            <span>120</span>
            <span>90</span>
            <span>60</span>
            <span>30</span>
            <span>10</span>
            <span>0</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PrecipitationNowcast;
