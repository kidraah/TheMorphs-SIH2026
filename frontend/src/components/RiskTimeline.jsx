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

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white border border-[#E2E2E2] shadow-md p-2 rounded text-[11px] font-sans">
        <p className="font-bold text-[#172033] border-b pb-1 mb-1">{label} IST Nowcast</p>
        {payload.map((item, index) => (
          <div key={index} className="flex items-center justify-between gap-3 py-0.5">
            <span className="font-semibold flex items-center gap-1.5" style={{ color: item.color }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: item.color }}></span>
              {item.name}:
            </span>
            <span className="font-extrabold text-[#172033]">{item.value}%</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export const RiskTimeline = () => {
  const { selectedTimeRange, riskTimeline } = useAppStore();

  // Load correct timeline based on selected time range (for Phase 10 prototype, we use the model's synthesized 6hr curve)
  const rawTimelineData = riskTimeline || [];
  
  // Filter for clean hourly ticks if there are many entries (to prevent crowding in a narrow column)
  const timelineData = rawTimelineData.length > 7
    ? rawTimelineData.filter((_, idx) => idx % 2 === 0)
    : rawTimelineData;

  // Label suffix based on range
  const rangeLabel = {
    now: '(1-Hr)',
    '2h': '(2-Hr)',
    '4h': '(4-Hr)',
    '6h': '(6-Hr)',
  }[selectedTimeRange] || '(6-Hr)';

  return (
    <div className="bg-white border border-[#E2E2E2] rounded p-3 shadow-gov-sm flex flex-col justify-between h-full min-h-[480px]">
      {/* Header */}
      <div className="border-b border-[#E2E2E2] pb-2 mb-2 flex items-center justify-between shrink-0">
        <h3 className="text-xs sm:text-[12.5px] font-bold text-[#172033] m-0 flex items-center gap-1.5">
          <span className="w-1 h-3.5 bg-[#E87516] rounded-sm inline-block shrink-0"></span>
          <span>Risk Timeline</span>
          <span className="text-slate-500 font-normal text-[10.5px]">{rangeLabel}</span>
        </h3>
      </div>

      {/* Recharts Multi-Series Line Chart - Fills available vertical height */}
      <div className="flex-1 w-full min-h-[340px] my-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timelineData} margin={{ top: 12, right: 10, left: -22, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 9.5, fill: '#64748B', fontWeight: 600 }}
              axisLine={{ stroke: '#CBD5E1' }}
              tickLine={{ stroke: '#CBD5E1' }}
            />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              tick={{ fontSize: 9.5, fill: '#64748B' }}
              axisLine={{ stroke: '#CBD5E1' }}
              tickLine={{ stroke: '#CBD5E1' }}
              label={{
                value: 'Risk Probability (%)',
                angle: -90,
                position: 'insideLeft',
                offset: 24,
                style: { fontSize: '8.5px', fill: '#64748B', textAnchor: 'middle', fontWeight: 600 },
              }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Thunderstorm (Purple) */}
            <Line
              type="monotone"
              dataKey="thunderstorm"
              name="Thunderstorm"
              stroke="#7E22CE"
              strokeWidth={2}
              dot={{ r: 3.5, fill: '#7E22CE', stroke: '#FFFFFF', strokeWidth: 1.5 }}
              activeDot={{ r: 5 }}
            />

            {/* Cloudburst (Red) */}
            <Line
              type="monotone"
              dataKey="cloudburst"
              name="Cloudburst"
              stroke="#DC2626"
              strokeWidth={2.2}
              dot={{ r: 3.5, fill: '#DC2626', stroke: '#FFFFFF', strokeWidth: 1.5 }}
              activeDot={{ r: 5 }}
            />

            {/* Flash Flood (Blue) */}
            <Line
              type="monotone"
              dataKey="flashFlood"
              name="Flash Flood"
              stroke="#0284C7"
              strokeWidth={2}
              dot={{ r: 3.5, fill: '#0284C7', stroke: '#FFFFFF', strokeWidth: 1.5 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Chart Legend */}
      <div className="flex items-center justify-around gap-2 pt-2 border-t border-slate-100 text-[10px] sm:text-[10.5px] shrink-0">
        <div className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-[#7E22CE] shrink-0"></span>
          <span className="text-[#172033] font-semibold">Thunderstorm</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-[#DC2626] shrink-0"></span>
          <span className="text-[#172033] font-semibold">Cloudburst</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-[#0284C7] shrink-0"></span>
          <span className="text-[#172033] font-semibold">Flash Flood</span>
        </div>
      </div>
    </div>
  );
};

export default RiskTimeline;
