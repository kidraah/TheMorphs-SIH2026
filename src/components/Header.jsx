import React from 'react';
import { ImdLogo, MoesLogo, Imd150Logo } from '../assets/emblems';
import govEmblem from '../assets/gov_emblem.png';
import pmHeaderImg from '../assets/prime-minister-header.png';

export const Header = () => {
  return (
    <header className="bg-white border-b border-slate-200">
      <div className="max-w-[1720px] mx-auto px-4 py-2 flex items-center justify-between gap-4">

        {/* ── LEFT: Ashoka Emblem Image + MoES label + IMD logo + IMD full name ── */}
        <div className="flex items-center gap-4">
          {/* National Emblem + Govt names */}
          <div className="flex items-center gap-2.5">
            <img
              src={govEmblem}
              alt="Government of India Emblem"
              className="h-[60px] w-auto object-contain shrink-0"
            />
            <div className="text-[10.5px] leading-[1.35] flex flex-col">
              <span className="font-bold text-slate-900">भारत सरकार</span>
              <span className="font-bold text-slate-700 text-[9.5px] uppercase tracking-wide">GOVERNMENT OF INDIA</span>
              <span className="font-semibold text-slate-700 mt-0.5">पृथ्वी विज्ञान मंत्रालय</span>
              <span className="font-semibold text-slate-500 text-[9px] uppercase tracking-wide">MINISTRY OF EARTH SCIENCES</span>
            </div>
          </div>

          {/* Thin divider */}
          <div className="h-10 w-px bg-slate-200 hidden sm:block" />

          {/* IMD Circular Crest + Department Title */}
          <div className="flex items-center gap-2.5">
            <ImdLogo className="h-[52px] w-auto" />
            <div>
              <div className="font-bold text-slate-900 text-[14px] leading-tight">
                भारत मौसम विज्ञान विभाग
              </div>
              <div className="font-extrabold text-[#0A3871] text-[12px] tracking-wide uppercase mt-0.5">
                INDIA METEOROLOGICAL DEPARTMENT
              </div>
            </div>
          </div>
        </div>

        {/* ── RIGHT: MoES logo + 150 Years logo + Prime Minister Header Asset ── */}
        <div className="flex items-center gap-3.5 ml-auto shrink-0">
          <div className="hidden md:block">
            <MoesLogo className="h-[44px] w-auto" />
          </div>
          <div className="h-9 w-px bg-slate-200 hidden md:block" />
          <Imd150Logo className="h-[44px] w-auto" />
          <div className="h-10 w-px bg-slate-200 hidden sm:block" />
          <img
            src={pmHeaderImg}
            alt="Prime Minister of India"
            className="h-[72px] md:h-[80px] w-auto object-contain shrink-0 hidden sm:block"
          />
        </div>
      </div>
    </header>
  );
};

export default Header;
