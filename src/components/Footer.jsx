import React from 'react';

export const Footer = () => {
  return (
    <footer className="bg-white border-t border-slate-200 mt-auto py-2.5 px-4 text-slate-600 text-xs">
      <div className="max-w-[1720px] mx-auto flex flex-col md:flex-row items-center justify-between gap-2 text-center md:text-left">
        <div className="flex items-center gap-1">
          <span>© {new Date().getFullYear()} India Meteorological Department</span>
        </div>
        <div className="font-semibold text-[#0A3871] hidden sm:block">
          AI-Driven Hyper-Local Early Warning System
        </div>
        <div className="text-slate-500 font-medium">
          Designed for Disaster Management Authorities
        </div>
      </div>
    </footer>
  );
};

export default Footer;
