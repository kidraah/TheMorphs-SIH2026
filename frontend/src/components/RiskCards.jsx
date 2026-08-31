import React from 'react';
import { RefreshCw, AlertTriangle, ShieldCheck } from 'lucide-react';
import { useCountdown } from '../hooks/useCountdown';
import { formatTime } from '../utils/formatters';
import { useAppStore } from '../store/useAppStore';

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

export const RiskCards = ({ layout = 'vertical' }) => {
  const { selectedDistrict, resetFilters, riskSummary } = useAppStore();
  const { secondsLeft, reset, isRefreshing } = useCountdown(930);

  // Fallback if data hasn't loaded yet
  const defaultSummary = {
    overallRisk: { level: 'Low', affectedDistricts: 0, gaugeValue: 0 },
    thunderstormRisk: { level: 'Low', probability: 0 },
    cloudburstRisk: { level: 'Low', probability: 0 },
    flashFloodRisk: { level: 'Low', probability: 0 },
  };

  const summaryForTime = riskSummary || defaultSummary;

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

  const containerClass = layout === 'horizontal'
    ? 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5'
    : 'flex flex-col gap-2 sm:gap-2.5 h-full justify-between';

  return (
    <div className={containerClass}>
      {/* 1. OVERALL RISK LEVEL */}
      <div className="bg-white border border-[#E2E2E2] rounded px-2.5 py-2 shadow-gov-sm flex items-center justify-between flex-1 min-h-[78px]">
        <div className="min-w-0">
          <span className="text-[9.5px] sm:text-[10px] font-bold text-slate-500 uppercase tracking-wide block mb-0.5">
            {isDistrictInspected ? 'District Risk Level' : 'Overall Risk Level'}
          </span>
          <span
            className="text-[17px] sm:text-[19px] font-extrabold leading-tight block"
            style={{ color: getRiskColor(displayData.overallRisk.level) }}
          >
            {displayData.overallRisk.level}
          </span>
          {isDistrictInspected ? (
            <button
              onClick={resetFilters}
              className="text-[9px] text-[#C85D00] font-bold hover:underline block mt-0.5 uppercase"
            >
              Clear: {displayData.overallRisk.affectedDistricts}
            </button>
          ) : (
            <span className="text-[9.5px] sm:text-[10px] text-slate-500 font-medium block mt-0.5 truncate">
              Across {displayData.overallRisk.affectedDistricts} Districts
            </span>
          )}
        </div>
      </div>

      {/* 2. THUNDERSTORM RISK */}
      <div className="bg-white border border-[#E2E2E2] rounded px-2.5 py-2 shadow-gov-sm flex items-center justify-between flex-1 min-h-[78px]">
        <div className="min-w-0">
          <span className="text-[9.5px] sm:text-[10px] font-bold text-slate-500 uppercase tracking-wide block mb-0.5">
            Thunderstorm Risk
          </span>
          <span
            className="text-[17px] sm:text-[19px] font-extrabold leading-tight block"
            style={{ color: getRiskColor(displayData.thunderstormRisk.level) }}
          >
            {displayData.thunderstormRisk.level}
          </span>
          <span className="text-[9.5px] sm:text-[10px] text-slate-500 font-medium block mt-0.5">
            Probability {displayData.thunderstormRisk.probability}%
          </span>
        </div>
      </div>

      {/* 3. CLOUDBURST RISK */}
      <div className="bg-white border border-[#E2E2E2] rounded px-2.5 py-2 shadow-gov-sm flex items-center justify-between flex-1 min-h-[78px]">
        <div className="min-w-0">
          <span className="text-[9.5px] sm:text-[10px] font-bold text-slate-500 uppercase tracking-wide block mb-0.5">
            Cloudburst Risk
          </span>
          <span
            className="text-[17px] sm:text-[19px] font-extrabold leading-tight block"
            style={{ color: getRiskColor(displayData.cloudburstRisk.level) }}
          >
            {displayData.cloudburstRisk.level}
          </span>
          <span className="text-[9.5px] sm:text-[10px] text-slate-500 font-medium block mt-0.5">
            Probability {displayData.cloudburstRisk.probability}%
          </span>
        </div>
      </div>

      {/* 4. FLASH FLOOD RISK */}
      <div className="bg-white border border-[#E2E2E2] rounded px-2.5 py-2 shadow-gov-sm flex items-center justify-between flex-1 min-h-[78px]">
        <div className="min-w-0">
          <span className="text-[9.5px] sm:text-[10px] font-bold text-slate-500 uppercase tracking-wide block mb-0.5">
            Flash Flood Risk
          </span>
          <span
            className="text-[17px] sm:text-[19px] font-extrabold leading-tight block"
            style={{ color: getRiskColor(displayData.flashFloodRisk.level) }}
          >
            {displayData.flashFloodRisk.level}
          </span>
          <span className="text-[9.5px] sm:text-[10px] text-slate-500 font-medium block mt-0.5">
            Probability {displayData.flashFloodRisk.probability}%
          </span>
        </div>
      </div>

      {/* 5. NEXT UPDATE IN */}
      <div className="bg-white border border-[#E2E2E2] rounded px-2.5 py-2 shadow-gov-sm flex items-center justify-between flex-1 min-h-[78px]">
        <div className="min-w-0">
          <span className="text-[9.5px] sm:text-[10px] font-bold text-slate-500 uppercase tracking-wide block mb-0.5">
            Next Update
          </span>
          <span
            className="font-mono font-extrabold text-emerald-600 leading-tight block"
            style={{ fontSize: '17px', letterSpacing: '0.03em' }}
          >
            {formatTime(secondsLeft)}
          </span>
          <span className="text-[9.5px] sm:text-[10px] text-slate-500 font-medium block mt-0.5">
            Stay Tuned
          </span>
        </div>
        <button
          type="button"
          onClick={reset}
          title="Refresh Nowcast"
          className="w-7 h-7 rounded-full border border-emerald-300 bg-emerald-50 hover:bg-emerald-100 flex items-center justify-center text-emerald-700 transition-colors shrink-0 ml-1.5 shadow-xs"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>
    </div>
  );
};

export default RiskCards;
