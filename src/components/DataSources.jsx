import React from 'react';
import { Satellite, Database, Radio, Map, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { mockDataSources } from '../data/mockData';

const getSourceIcon = (id) => {
  switch (id) {
    case 'insat':
      return <Satellite className="w-3.5 h-3.5 text-slate-600 shrink-0" />;
    case 'imdaa':
      return <Database className="w-3.5 h-3.5 text-slate-600 shrink-0" />;
    case 'qpe':
      return <Radio className="w-3.5 h-3.5 text-slate-600 shrink-0" />;
    case 'dem':
    default:
      return <Map className="w-3.5 h-3.5 text-slate-600 shrink-0" />;
  }
};

export const DataSources = ({ sources = mockDataSources }) => {
  return (
    <div className="bg-white border border-[#E2E2E2] rounded p-3 shadow-gov-sm flex flex-col justify-between h-full">
      {/* Header */}
      <div className="border-b border-[#E2E2E2] pb-1.5 mb-1.5">
        <h3 className="text-xs sm:text-[13px] font-bold text-[#172033] m-0 flex items-center gap-1.5">
          <span className="w-1 h-3.5 bg-[#E87516] rounded-sm inline-block"></span>
          <span>Data Sources</span>
        </h3>
      </div>

      {/* Source Items */}
      <div className="space-y-1.5">
        {sources.map((item) => (
          <div
            key={item.id}
            className="flex items-center justify-between gap-2 py-0.5"
          >
            <div className="flex items-center gap-2 min-w-0">
              {getSourceIcon(item.id)}
              <span className="text-[11.5px] font-medium text-slate-800 truncate">
                {item.name}
              </span>
            </div>
            <span
              className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                item.type === 'live'
                  ? 'text-emerald-700 bg-emerald-50 border border-emerald-200'
                  : 'text-slate-600 bg-slate-100 border border-slate-200'
              }`}
            >
              {item.status}
            </span>
          </div>
        ))}
      </div>

      {/* Footer Link */}
      <div className="pt-1.5 border-t border-slate-100 mt-2">
        <Link
          to="/data-layers"
          className="inline-flex items-center gap-1 text-[11px] font-bold text-[#C85D00] hover:text-[#A84A00] hover:underline"
        >
          <span>View All Sources</span>
          <ArrowRight className="w-3 h-3" />
        </Link>
      </div>
    </div>
  );
};

export default DataSources;
