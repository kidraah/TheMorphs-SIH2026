import React from 'react';
import { ImdLogo, MoesLogo, Imd150Logo } from '../assets/emblems';
import govEmblem from '../assets/gov_emblem.png';
import ndmaLogo from '../assets/ndma_logo.png';

export const Header = () => {
  return (
    <header className="bg-white border-b border-[#E2E2E2]">
      <div className="max-w-[1720px] mx-auto px-4 py-2.5 sm:py-3 flex items-center justify-between gap-4">

        {/* ── LEFT: Ashoka Emblem Image + MoES label + IMD logo + IMD full name ── */}
        <div className="flex items-center gap-4 sm:gap-5">
          {/* National Emblem + Govt names */}
          <div className="flex items-center gap-3">
            <img
              src={govEmblem}
              alt="Government of India Emblem"
              className="h-[96px] md:h-[110px] w-auto object-contain shrink-0"
            />
            <div className="leading-[1.3] flex flex-col justify-center">
              <span className="font-bold text-slate-900 text-[14px] md:text-[16px]">भारत सरकार</span>
              <span className="font-bold text-slate-700 text-[12px] md:text-[13px] uppercase tracking-wider">GOVERNMENT OF INDIA</span>
              <span className="font-semibold text-slate-700 text-[13px] md:text-[14px] mt-0.5">पृथ्वी विज्ञान मंत्रालय</span>
              <span className="font-semibold text-slate-500 text-[11px] md:text-[12px] uppercase tracking-wider">MINISTRY OF EARTH SCIENCES</span>
            </div>
          </div>

          {/* Thin divider */}
          <div className="h-12 w-px bg-slate-200 hidden sm:block" />

          {/* IMD Circular Crest + Department Title */}
          <div className="flex items-center gap-3">
            <ImdLogo className="h-[80px] md:h-[90px] w-auto shrink-0" />
            <div>
              <div className="font-bold text-slate-900 text-[18px] md:text-[20px] leading-tight">
                भारत मौसम विज्ञान विभाग
              </div>
              <div className="font-extrabold text-[#E87516] text-[15px] md:text-[16px] tracking-wide uppercase mt-0.5">
                INDIA METEOROLOGICAL DEPARTMENT
              </div>
            </div>
          </div>
        </div>

        {/* ── RIGHT: NDMA Logo + MoES Logo + 150 Years Logo ── */}
        <div className="flex items-center gap-4 md:gap-5 ml-auto shrink-0">
          {/* NDMA Logo */}
          <img
            src={ndmaLogo}
            alt="National Disaster Management Authority (NDMA)"
            className="h-[96px] md:h-[110px] w-auto object-contain shrink-0"
          />
          <div className="h-16 w-px bg-slate-200 hidden sm:block" />

          {/* MoES Logo */}
          <div className="hidden md:block">
            <MoesLogo className="h-[70px] md:h-[80px] w-auto" />
          </div>
          <div className="h-16 w-px bg-slate-200 hidden md:block" />

          {/* 150 Years Logo */}
          <Imd150Logo className="h-[70px] md:h-[80px] w-auto" />
        </div>
      </div>
    </header>
  );
};

export default Header;
