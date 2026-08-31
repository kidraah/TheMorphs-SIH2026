import React, { useState } from 'react';
import { Search, ChevronDown, Menu, X, Bell, MapPin, Clock, LayoutDashboard, Map, TrendingUp, CloudSun, Layers, FileText, Settings, Info } from 'lucide-react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import { useAppStore } from '../store/useAppStore';

export const Navbar = () => {
  const location = useLocation();
  const {
    searchQuery, setSearchQuery,
    alerts, acknowledgedAlerts,
    selectedRegion, setSelectedRegion, regions,
    selectedTimeRange, setSelectedTimeRange,
    fetchDashboardData,
  } = useAppStore();

  const [activeDropdown, setActiveDropdown] = useState(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const unreadAlertCount = alerts
    ? alerts.filter((a) => !acknowledgedAlerts.includes(a.id)).length
    : 0;

  const timeOptions = [
    { id: 'now', label: 'Now' },
    { id: '2h', label: '2 Hours' },
    { id: '4h', label: '4 Hours' },
    { id: '6h', label: '6 Hours' },
  ];

  return (
    <div className="w-full font-sans">
      {/* ── 1. Primary Saffron Navigation Bar ────────────────────── */}
      <nav
        className="bg-[#E87516] text-white select-none relative z-[1000]"
        onMouseLeave={() => setActiveDropdown(null)}
      >
        <div className="max-w-[1720px] mx-auto px-3 sm:px-4 flex items-center justify-between min-h-[56px]">
          
          {/* Mobile Hamburger Toggle Button */}
          <div className="flex items-center lg:hidden py-1.5">
            <button
              type="button"
              onClick={() => setMobileMenuOpen((prev) => !prev)}
              className="p-1.5 rounded hover:bg-[#C85D00] text-white transition-colors flex items-center gap-1.5 text-xs font-bold"
              aria-label="Toggle navigation menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              <span>MENU</span>
            </button>
          </div>

          {/* Desktop & Tablet Navigation Links */}
          <div className="hidden lg:flex items-center space-x-0.5 overflow-x-auto scrollbar-none">
            {/* DASHBOARD */}
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `inline-flex items-center px-4 py-4 text-[14px] font-bold tracking-wider uppercase transition-colors whitespace-nowrap ${
                  isActive && location.pathname === '/'
                    ? 'bg-[#C85D00] text-white border-b-2 border-white'
                    : 'text-white hover:bg-[#C85D00]'
                }`
              }
            >
              DASHBOARD
            </NavLink>

            {/* LIVE MAP */}
            <NavLink
              to="/live-map"
              className={({ isActive }) =>
                `inline-flex items-center px-4 py-4 text-[14px] font-bold tracking-wider uppercase transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-[#C85D00] text-white border-b-2 border-white'
                    : 'text-white hover:bg-[#C85D00]'
                }`
              }
            >
              LIVE MAP
            </NavLink>

            {/* RISK OUTLOOK */}
            <NavLink
              to="/risk-outlook"
              className={({ isActive }) =>
                `inline-flex items-center px-4 py-4 text-[14px] font-bold tracking-wider uppercase transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-[#C85D00] text-white border-b-2 border-white'
                    : 'text-white hover:bg-[#C85D00]'
                }`
              }
            >
              RISK OUTLOOK
            </NavLink>

            {/* ALERTS WITH RED COUNT BADGE */}
            <NavLink
              to="/alerts"
              className={({ isActive }) =>
                `inline-flex items-center gap-1.5 px-4 py-4 text-[14px] font-bold tracking-wider uppercase transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-[#C85D00] text-white border-b-2 border-white'
                    : 'text-white hover:bg-[#C85D00]'
                }`
              }
            >
              <span>ALERTS</span>
              {unreadAlertCount > 0 && (
                <span className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 text-[10px] font-extrabold text-white bg-red-600 border border-white rounded-full leading-none animate-pulse">
                  {unreadAlertCount}
                </span>
              )}
            </NavLink>

            {/* FORECAST */}
            <NavLink
              to="/forecast"
              className={({ isActive }) =>
                `inline-flex items-center px-4 py-4 text-[14px] font-bold tracking-wider uppercase transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-[#C85D00] text-white border-b-2 border-white'
                    : 'text-white hover:bg-[#C85D00]'
                }`
              }
            >
              FORECAST
            </NavLink>

            {/* DATA LAYERS */}
            <NavLink
              to="/data-layers"
              className={({ isActive }) =>
                `inline-flex items-center px-4 py-4 text-[14px] font-bold tracking-wider uppercase transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-[#C85D00] text-white border-b-2 border-white'
                    : 'text-white hover:bg-[#C85D00]'
                }`
              }
            >
              DATA LAYERS
            </NavLink>

            {/* REPORTS */}
            <NavLink
              to="/reports"
              className={({ isActive }) =>
                `inline-flex items-center px-4 py-4 text-[14px] font-bold tracking-wider uppercase transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-[#C85D00] text-white border-b-2 border-white'
                    : 'text-white hover:bg-[#C85D00]'
                }`
              }
            >
              REPORTS
            </NavLink>


          </div>

          {/* Search Bar & Quick Indicators */}
          <div className="flex items-center py-1 shrink-0 pl-2">
            <div className="relative flex items-center">
              <input
                type="text"
                placeholder="Search districts, alerts..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-36 sm:w-48 h-[28px] px-2.5 pr-7 text-[11.5px] bg-white text-[#172033] rounded-l border border-orange-200 focus:outline-none focus:ring-1 focus:ring-white"
              />
              <button
                type="button"
                className="h-[28px] px-2.5 bg-[#C85D00] hover:bg-[#A84A00] text-white rounded-r border border-[#C85D00] flex items-center justify-center transition-colors"
              >
                <Search className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Hamburger Drawer Menu */}
        {mobileMenuOpen && (
          <div className="lg:hidden bg-white text-[#172033] border-b-2 border-[#E87516] shadow-2xl p-3 space-y-1">
            <NavLink
              to="/"
              end
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded text-xs font-bold transition-colors ${
                  isActive ? 'bg-[#FFF1DD] text-[#C85D00] border-l-4 border-[#E87516]' : 'text-slate-800 hover:bg-slate-50'
                }`
              }
            >
              <LayoutDashboard className="w-4 h-4" />
              <span>Dashboard</span>
            </NavLink>

            <NavLink
              to="/live-map"
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded text-xs font-bold transition-colors ${
                  isActive ? 'bg-[#FFF1DD] text-[#C85D00] border-l-4 border-[#E87516]' : 'text-slate-800 hover:bg-slate-50'
                }`
              }
            >
              <Map className="w-4 h-4" />
              <span>Live Risk Map</span>
            </NavLink>

            <NavLink
              to="/risk-outlook"
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded text-xs font-bold transition-colors ${
                  isActive ? 'bg-[#FFF1DD] text-[#C85D00] border-l-4 border-[#E87516]' : 'text-slate-800 hover:bg-slate-50'
                }`
              }
            >
              <TrendingUp className="w-4 h-4" />
              <span>Risk Outlook</span>
            </NavLink>

            <NavLink
              to="/alerts"
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) =>
                `flex items-center justify-between px-3 py-2 rounded text-xs font-bold transition-colors ${
                  isActive ? 'bg-[#FFF1DD] text-[#C85D00] border-l-4 border-[#E87516]' : 'text-slate-800 hover:bg-slate-50'
                }`
              }
            >
              <div className="flex items-center gap-2">
                <Bell className="w-4 h-4" />
                <span>Alerts</span>
              </div>
              {unreadAlertCount > 0 && (
                <span className="px-2 py-0.5 text-[10px] font-bold text-white bg-red-600 rounded-full">
                  {unreadAlertCount}
                </span>
              )}
            </NavLink>

            <NavLink
              to="/forecast"
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded text-xs font-bold transition-colors ${
                  isActive ? 'bg-[#FFF1DD] text-[#C85D00] border-l-4 border-[#E87516]' : 'text-slate-800 hover:bg-slate-50'
                }`
              }
            >
              <CloudSun className="w-4 h-4" />
              <span>Forecast</span>
            </NavLink>

            <NavLink
              to="/data-layers"
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded text-xs font-bold transition-colors ${
                  isActive ? 'bg-[#FFF1DD] text-[#C85D00] border-l-4 border-[#E87516]' : 'text-slate-800 hover:bg-slate-50'
                }`
              }
            >
              <Layers className="w-4 h-4" />
              <span>Data Layers</span>
            </NavLink>

            <NavLink
              to="/reports"
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded text-xs font-bold transition-colors ${
                  isActive ? 'bg-[#FFF1DD] text-[#C85D00] border-l-4 border-[#E87516]' : 'text-slate-800 hover:bg-slate-50'
                }`
              }
            >
              <FileText className="w-4 h-4" />
              <span>Reports</span>
            </NavLink>

            <NavLink
              to="/settings"
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded text-xs font-bold transition-colors ${
                  isActive ? 'bg-[#FFF1DD] text-[#C85D00] border-l-4 border-[#E87516]' : 'text-slate-800 hover:bg-slate-50'
                }`
              }
            >
              <Settings className="w-4 h-4" />
              <span>Settings</span>
            </NavLink>

            <NavLink
              to="/about-system"
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded text-xs font-bold transition-colors ${
                  isActive ? 'bg-[#FFF1DD] text-[#C85D00] border-l-4 border-[#E87516]' : 'text-slate-800 hover:bg-slate-50'
                }`
              }
            >
              <Info className="w-4 h-4" />
              <span>About System</span>
            </NavLink>
          </div>
        )}
      </nav>

      {/* ── 2. Sub-Header: Title Banner + Region & Time Controls ── */}
      <div className="bg-white border-b border-[#E2E2E2] py-2 px-3 sm:px-4">
        <div className="max-w-[1720px] mx-auto flex flex-col md:flex-row items-center justify-between gap-2.5">
          
          {/* Left: System Identification Title */}
          <div className="text-center md:text-left">
            <p className="text-[14px] sm:text-[15px] font-extrabold text-[#172033] tracking-wide m-0 leading-tight">
              AI-Driven Hyper-Local Early Warning System
            </p>
            <p className="text-[11.5px] sm:text-[12px] font-semibold text-slate-600 mt-0.5 m-0">
              Nowcasting for Severe Thunderstorm, Cloudburst &amp; Flash Flood
            </p>
          </div>

          {/* Right: Operational Controls Bar (Region & Time Range) */}
          <div className="flex items-center flex-wrap gap-2.5 justify-center md:justify-end text-xs">
            {/* Region Dropdown */}
            <div className="flex items-center gap-1.5 bg-[#FFF7EA] border border-orange-200 px-2 py-1 rounded shadow-sm">
              <MapPin className="w-3.5 h-3.5 text-[#E87516] shrink-0" />
              <span className="text-[10.5px] font-bold text-[#172033] uppercase">Region:</span>
              <select
                value={selectedRegion}
                onChange={(e) => setSelectedRegion(e.target.value)}
                className="bg-white text-[11px] font-semibold text-[#172033] border border-orange-300 rounded px-1.5 py-0.5 cursor-pointer focus:outline-none focus:ring-1 focus:ring-[#E87516]"
              >
                {regions.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Time Range Selector */}
            <div className="flex items-center gap-1 bg-slate-50 border border-slate-200 px-1.5 py-1 rounded shadow-sm">
              <Clock className="w-3.5 h-3.5 text-slate-500 shrink-0 ml-0.5" />
              <span className="text-[10.5px] font-bold text-slate-600 uppercase mr-1">Time:</span>
              <div className="flex items-center gap-1">
                {timeOptions.map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => {
                      setSelectedTimeRange(opt.id);
                      fetchDashboardData(opt.id);
                    }}
                    className={`px-2 py-0.5 text-[10.5px] rounded font-semibold transition-all ${
                      selectedTimeRange === opt.id
                        ? 'bg-[#E87516] text-white font-bold shadow-xs'
                        : 'bg-white text-slate-700 hover:bg-slate-100 border border-slate-200'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default Navbar;
