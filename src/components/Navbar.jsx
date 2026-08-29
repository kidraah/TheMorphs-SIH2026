import React, { useState } from 'react';
import { Search, ChevronDown } from 'lucide-react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { useAppStore } from '../store/useAppStore';

const NAV_LINKS = [
  { label: 'HOME', path: '/' },
  { label: 'ABOUT IMD', dropdown: ['Organization Overview', 'Mandate & Vision', 'Observational Network', 'Citizen Charter'] },
  { label: 'WEATHER', dropdown: ['Nowcast Warnings', 'Severe Weather Outlook', 'Radar Imagery', 'Monsoon Tracking'] },
  { label: 'CLIMATE', dropdown: ['Climate Diagnostic', 'Rainfall Statistics', 'Drought Monitoring', 'ENSO / IOD'] },
  { label: 'DATA SUPPLY', path: '/data-layers' },
  { label: 'RESEARCH', path: '/forecast' },
  { label: 'PUBLICATIONS', dropdown: ['Mausam Journal', 'Technical Reports', 'Monsoon Monographs', 'Special Bulletins'] },
  { label: 'SOP', dropdown: ['Disaster SOPs', 'Cyclone Warning SOP', 'Flash Flood Protocol', 'State Nodal Officers'] },
  { label: 'SERVICES', dropdown: ['Agromet Services', 'Aviation Meteorology', 'Hydromet Warnings', 'Tourism Forecast'] },
  { label: 'PRESS RELEASE', path: '/reports' },
  { label: 'CONTACTS', path: '/about-system' },
  { label: 'LOCAL FORECAST', path: '/live-map' },
];

export const Navbar = () => {
  const location = useLocation();
  const { searchQuery, setSearchQuery } = useAppStore();
  const [activeDropdown, setActiveDropdown] = useState(null);

  return (
    <div className="w-full">
      {/* Primary Navy Blue Navigation Bar */}
      <nav
        className="bg-[#0A3871] text-white"
        onMouseLeave={() => setActiveDropdown(null)}
      >
        <div className="max-w-[1720px] mx-auto px-3 flex items-center justify-between">
          {/* Nav Links Row */}
          <div className="flex items-center overflow-x-auto scrollbar-none">
            {NAV_LINKS.map((item, idx) => (
              <div
                key={idx}
                className="relative shrink-0"
                onMouseEnter={() => item.dropdown ? setActiveDropdown(idx) : setActiveDropdown(null)}
              >
                {item.path ? (
                  <NavLink
                    to={item.path}
                    end={item.path === '/'}
                    className={({ isActive }) =>
                      `inline-flex items-center px-2.5 py-2.5 text-[11.5px] font-semibold tracking-wide uppercase transition-colors ${
                        isActive
                          ? 'bg-[#06254F] text-amber-300 border-b-2 border-amber-400'
                          : 'text-slate-100 hover:bg-[#134D96]'
                      }`
                    }
                  >
                    {item.label}
                  </NavLink>
                ) : (
                  <button
                    type="button"
                    className="inline-flex items-center gap-0.5 px-2.5 py-2.5 text-[11.5px] font-semibold tracking-wide uppercase text-slate-100 hover:bg-[#134D96] transition-colors"
                    onClick={() => setActiveDropdown(activeDropdown === idx ? null : idx)}
                  >
                    {item.label}
                    <ChevronDown className="w-3 h-3 text-slate-300 ml-0.5" />
                  </button>
                )}

                {/* Dropdown Panel */}
                {item.dropdown && activeDropdown === idx && (
                  <div className="absolute left-0 top-full w-52 bg-white text-slate-800 shadow-xl border border-slate-200 z-[1000] py-1">
                    {item.dropdown.map((sub, sIdx) => (
                      <a
                        key={sIdx}
                        href="#"
                        onClick={(e) => { e.preventDefault(); setActiveDropdown(null); }}
                        className="block px-3.5 py-1.5 text-[11px] text-slate-700 hover:bg-blue-50 hover:text-[#0A3871] font-medium transition-colors"
                      >
                        {sub}
                      </a>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Search Bar */}
          <div className="flex items-center py-1 shrink-0 pl-3">
            <div className="relative flex items-center">
              <input
                type="text"
                placeholder="Search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-40 h-[28px] px-2.5 pr-7 text-[11.5px] bg-white text-slate-800 rounded-l border border-slate-300 focus:outline-none focus:ring-1 focus:ring-blue-400"
              />
              <button
                type="button"
                className="h-[28px] px-2 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-r border border-l-0 border-slate-300 flex items-center justify-center"
              >
                <Search className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Sub-Header: System Title Banner */}
      <div className="bg-white border-b border-slate-200 py-2 text-center">
        <p className="text-[15px] sm:text-base font-extrabold text-[#CC0000] tracking-wide m-0 leading-tight">
          AI-Driven Hyper-Local Early Warning System
        </p>
        <p className="text-[12px] sm:text-[13px] font-semibold text-[#991B1B] mt-0.5 m-0">
          Nowcasting for Severe Thunderstorm, Cloudburst &amp; Flash Flood
        </p>
      </div>
    </div>
  );
};

export default Navbar;
