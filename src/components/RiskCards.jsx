import React from 'react';
import { RefreshCw } from 'lucide-react';
import { useCountdown } from '../hooks/useCountdown';
import { formatTime } from '../utils/formatters';
import { useAppStore } from '../store/useAppStore';
import { mockRiskSummaryByTime } from '../data/mockData';

const getRiskLevelName = (prob) => {
  if (prob >= 86) return 'Extreme';
  if (prob >= 76) return 'Very High';
  if (prob >= 56) return 'High';
  if (prob >= 36) return 'Moderate';
  return 'Low';
};

const getRiskColor = (level) => {
  switch (level.toLowerCase()) {
    case 'extreme': return '#7E22CE';
    case 'very high': return '#DC2626';
    case 'high': return '#EA580C';
    case 'moderate': return '#EAB308';
    default: return '#16A34A';
  }
};

export const RiskCards = () => {
  const { selectedTimeRange, selectedDistrict, resetFilters } = useAppStore();
  const { secondsLeft, reset, isRefreshing } = useCountdown(930);

  // Fallback to base summary based on selected time range
  const summaryForTime = mockRiskSummaryByTime[selectedTimeRange] || mockRiskSummaryByTime['2h'];

  let displayData = summaryForTime;
  let isDistrictInspected = false;

  if (selectedDistrict) {
    isDistrictInspected = true;
    const overallLevel = selectedDistrict.riskLevel;
    const tProb = selectedDistrict.thunderstormProbability;
    const cProb = selectedDistrict.cloudburstProbability;
    const fProb = selectedDistrict.flashFloodProbability;
    const maxProb = Math.max(tProb, cProb, fProb);

    displayData = {
      overallRisk: {
        level: overallLevel.charAt(0).toUpperCase() + overallLevel.slice(1).toLowerCase(),
        affectedDistricts: selectedDistrict.district,
        gaugeValue: maxProb,
      },
      thunderstormRisk: {
        level: getRiskLevelName(tProb),
        probability: tProb,
      },
      cloudburstRisk: {
        level: getRiskLevelName(cProb),
        probability: cProb,
      },
      flashFloodRisk: {
        level: getRiskLevelName(fProb),
        probability: fProb,
      },
    };
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
      {/* 1. Overall Risk Level */}
      <div className="bg-white border border-slate-200 rounded p-3 shadow-sm flex items-center justify-between">
        <div className="min-w-0">
          <span className="text-[10.5px] font-bold text-slate-500 uppercase tracking-wide block mb-0.5">
            {isDistrictInspected ? 'District Risk Level' : 'Overall Risk Level'}
          </span>
          <span
            className="text-[22px] font-extrabold leading-none block"
            style={{ color: getRiskColor(displayData.overallRisk.level) }}
          >
            {displayData.overallRisk.level}
          </span>
          {isDistrictInspected ? (
            <button
              onClick={resetFilters}
              className="text-[9.5px] text-[#C85D00] font-bold hover:underline block mt-1 uppercase"
            >
              Clear: {displayData.overallRisk.affectedDistricts}
            </button>
          ) : (
            <span className="text-[10.5px] text-slate-500 font-medium block mt-0.5">
              Across {displayData.overallRisk.affectedDistricts} Districts
            </span>
          )}
        </div>
      </div>

      {/* 2. Thunderstorm Risk */}
      <div className="bg-white border border-slate-200 rounded p-3 shadow-sm flex items-center justify-between">
        <div className="min-w-0">
          <span className="text-[10.5px] font-bold text-slate-500 uppercase tracking-wide block mb-0.5">
            Thunderstorm Risk
          </span>
          <span
            className="text-[22px] font-extrabold leading-none block"
            style={{ color: getRiskColor(displayData.thunderstormRisk.level) }}
          >
            {displayData.thunderstormRisk.level}
          </span>
          <span className="text-[10.5px] text-slate-500 font-medium block mt-0.5">
            Probability {displayData.thunderstormRisk.probability}%
          </span>
        </div>
      </div>

      {/* 3. Cloudburst Risk */}
      <div className="bg-white border border-slate-200 rounded p-3 shadow-sm flex items-center justify-between">
        <div className="min-w-0">
          <span className="text-[10.5px] font-bold text-slate-500 uppercase tracking-wide block mb-0.5">
            Cloudburst Risk
          </span>
          <span
            className="text-[22px] font-extrabold leading-none block"
            style={{ color: getRiskColor(displayData.cloudburstRisk.level) }}
          >
            {displayData.cloudburstRisk.level}
          </span>
          <span className="text-[10.5px] text-slate-500 font-medium block mt-0.5">
            Probability {displayData.cloudburstRisk.probability}%
          </span>
        </div>
      </div>

      {/* 4. Flash Flood Risk */}
      <div className="bg-white border border-slate-200 rounded p-3 shadow-sm flex items-center justify-between">
        <div className="min-w-0">
          <span className="text-[10.5px] font-bold text-slate-500 uppercase tracking-wide block mb-0.5">
            Flash Flood Risk
          </span>
          <span
            className="text-[22px] font-extrabold leading-none block"
            style={{ color: getRiskColor(displayData.flashFloodRisk.level) }}
          >
            {displayData.flashFloodRisk.level}
          </span>
          <span className="text-[10.5px] text-slate-500 font-medium block mt-0.5">
            Probability {displayData.flashFloodRisk.probability}%
          </span>
        </div>
      </div>

      {/* 5. Next Update In (Countdown Timer) */}
      <div className="bg-white border border-slate-200 rounded p-3 shadow-sm flex items-center justify-between">
        <div className="min-w-0">
          <span className="text-[10.5px] font-bold text-slate-500 uppercase tracking-wide block mb-0.5">
            Next Update In
          </span>
          <span
            className="font-mono font-extrabold text-emerald-600 leading-none block"
            style={{ fontSize: '20px', letterSpacing: '0.04em' }}
          >
            {formatTime(secondsLeft)}
          </span>
          <span className="text-[10.5px] text-slate-500 font-medium block mt-0.5">
            Stay Tuned
          </span>
        </div>
        <button
          type="button"
          onClick={reset}
          title="Refresh Nowcast"
          className="w-10 h-10 rounded-full border border-emerald-300 bg-emerald-50 hover:bg-emerald-100 flex items-center justify-center text-emerald-700 transition-colors shrink-0 ml-2"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>
    </div>
  );
};

export default RiskCards;
