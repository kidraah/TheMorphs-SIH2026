import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { useAppStore } from '../store/useAppStore';
import { mockRiskTimelineByTime } from '../data/mockData';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white border border-slate-300 shadow-md p-2 rounded text-[11px] font-sans">
        <p className="font-bold text-slate-900 border-b pb-1 mb-1">{label} IST Nowcast</p>
        {payload.map((item, index) => (
          <div key={index} className="flex items-center justify-between gap-3 py-0.5">
            <span className="font-medium flex items-center gap-1.5" style={{ color: item.color }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: item.color }}></span>
              {item.name}:
            </span>
            <span className="font-bold text-slate-900">{item.value}%</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export const RiskTimeline = () => {
  const { selectedTimeRange } = useAppStore();

  // Load correct timeline based on selected time range
  const timelineData = mockRiskTimelineByTime[selectedTimeRange] || mockRiskTimelineByTime['6h'];

  // Label suffix based on range
  const rangeLabel = {
    now: '(Next Hour Nowcast)',
    '2h': '(Next 2 Hours)',
    '4h': '(Next 4 Hours)',
    '6h': '(Next 6 Hours)',
  }[selectedTimeRange] || '(Next 6 Hours)';

  return (
    <div className="bg-white border border-[#E2E2E2] rounded p-3 shadow-gov-sm flex flex-col justify-between h-full">
      {/* Header */}
      <div className="border-b border-[#E2E2E2] pb-1.5 mb-1.5 flex items-center justify-between">
        <h3 className="text-xs sm:text-[13px] font-bold text-[#172033] m-0 flex items-center gap-1.5">
          <span className="w-1 h-3.5 bg-[#E87516] rounded-sm inline-block"></span>
          <span>Risk Timeline</span>
          <span className="text-slate-500 font-normal text-[11px]">{rangeLabel}</span>
        </h3>
      </div>

      {/* Recharts Multi-Series Line Chart */}
      <div className="h-40 sm:h-44 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timelineData} margin={{ top: 8, right: 12, left: -22, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 10, fill: '#64748B' }}
              axisLine={{ stroke: '#CBD5E1' }}
              tickLine={{ stroke: '#CBD5E1' }}
            />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              tick={{ fontSize: 10, fill: '#64748B' }}
              axisLine={{ stroke: '#CBD5E1' }}
              tickLine={{ stroke: '#CBD5E1' }}
              label={{
                value: 'Risk Probability (%)',
                angle: -90,
                position: 'insideLeft',
                offset: 24,
                style: { fontSize: '9px', fill: '#64748B', textAnchor: 'middle' },
              }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Thunderstorm (Purple) */}
            <Line
              type="monotone"
              dataKey="thunderstorm"
              name="Thunderstorm"
              stroke="#7E22CE"
              strokeWidth={1.8}
              dot={{ r: 3, fill: '#7E22CE', stroke: '#FFFFFF', strokeWidth: 1 }}
              activeDot={{ r: 4.5 }}
            />

            {/* Cloudburst (Red) */}
            <Line
              type="monotone"
              dataKey="cloudburst"
              name="Cloudburst"
              stroke="#DC2626"
              strokeWidth={2}
              dot={{ r: 3, fill: '#DC2626', stroke: '#FFFFFF', strokeWidth: 1 }}
              activeDot={{ r: 4.5 }}
            />

            {/* Flash Flood (Blue) */}
            <Line
              type="monotone"
              dataKey="flashFlood"
              name="Flash Flood"
              stroke="#0284C7"
              strokeWidth={1.8}
              dot={{ r: 3, fill: '#0284C7', stroke: '#FFFFFF', strokeWidth: 1 }}
              activeDot={{ r: 4.5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Chart Legend */}
      <div className="flex items-center justify-center gap-4 pt-1.5 border-t border-slate-100 text-[10.5px]">
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#7E22CE]"></span>
          <span className="text-slate-700 font-medium">Thunderstorm</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#DC2626]"></span>
          <span className="text-slate-700 font-medium">Cloudburst</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#0284C7]"></span>
          <span className="text-slate-700 font-medium">Flash Flood</span>
        </div>
      </div>
    </div>
  );
};

export default RiskTimeline;
