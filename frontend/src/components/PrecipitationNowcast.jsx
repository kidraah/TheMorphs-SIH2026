import React from 'react';
import { useAppStore } from '../store/useAppStore';

export const PrecipitationNowcast = () => {
  const { radarImage } = useAppStore();

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
        {/* Radar Heatmap Image */}
        <div className="relative flex-1 h-full bg-[#0F284E] rounded border border-slate-300 overflow-hidden">
          {radarImage ? (
            <img src={radarImage} alt="Radar Nowcast" className="w-full h-full object-cover" />
          ) : (
            <div className="flex items-center justify-center w-full h-full text-slate-400 text-[10px] font-medium">
              <span className="animate-pulse">Loading Radar...</span>
            </div>
          )}
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
