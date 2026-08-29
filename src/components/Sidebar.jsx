import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Map,
  TrendingUp,
  Bell,
  CloudSun,
  Layers,
  FileText,
  Settings,
  Info,
  CheckCircle2,
} from 'lucide-react';
import RegionSelector from './RegionSelector';
import TimeRangeSelector from './TimeRangeSelector';
import { useAppStore } from '../store/useAppStore';

export const Sidebar = () => {
  const { alerts } = useAppStore();
  const unreadCount = alerts.filter
    ? alerts.filter((a, _, arr) => true).length
    : 5;

  const menuItems = [
    { name: 'Dashboard', path: '/', icon: LayoutDashboard },
    { name: 'Live Map', path: '/live-map', icon: Map },
    { name: 'Risk Outlook', path: '/risk-outlook', icon: TrendingUp },
    { name: 'Alerts', path: '/alerts', icon: Bell, badge: unreadCount },
    { name: 'Forecast', path: '/forecast', icon: CloudSun },
    { name: 'Data Layers', path: '/data-layers', icon: Layers },
    { name: 'Reports', path: '/reports', icon: FileText },
    { name: 'Settings', path: '/settings', icon: Settings },
    { name: 'About System', path: '/about-system', icon: Info },
  ];

  return (
    <aside className="w-full lg:w-[220px] shrink-0 bg-white border-r border-slate-200 flex flex-col min-h-full">
      {/* Navigation Links */}
      <nav className="p-2 space-y-0.5 flex-1">
        {menuItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.name}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                `flex items-center justify-between px-3 py-2 rounded text-xs font-semibold transition-colors duration-100 border-l-4 ${
                  isActive
                    ? 'bg-[#EBF3FC] text-[#1D4ED8] border-[#1D4ED8]'
                    : 'text-slate-700 hover:bg-slate-50 hover:text-slate-900 border-transparent'
                }`
              }
            >
              <div className="flex items-center gap-2">
                <Icon className="w-[15px] h-[15px] shrink-0" />
                <span>{item.name}</span>
              </div>
              {item.badge && (
                <span className="inline-flex items-center justify-center w-4 h-4 text-[9.5px] font-bold text-white bg-red-600 rounded-full">
                  {item.badge}
                </span>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Divider */}
      <div className="h-px bg-slate-200 mx-3"></div>

      {/* Region & Time Range Controls */}
      <div className="p-2.5 space-y-2">
        <RegionSelector />
        <TimeRangeSelector />
      </div>

      {/* Divider */}
      <div className="h-px bg-slate-200 mx-3"></div>

      {/* System Status Footer */}
      <div className="p-2.5">
        <div className="bg-emerald-50 border border-emerald-200 rounded p-2.5">
          <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block mb-1">
            System Status
          </span>
          <div className="flex items-center gap-2 text-emerald-800">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            <span className="text-[11.5px] font-bold leading-tight">All Systems Operational</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
