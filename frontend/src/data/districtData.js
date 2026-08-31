/**
 * Backward-compatible re-export of district risk data.
 *
 * Maps the new canonical fields from districtRiskData.js
 * to the legacy field names used by LiveMapPage, RiskOutlookPage,
 * useAppStore, and apiService.
 */

import { DISTRICT_RISK_DATA as _CANONICAL } from './districtRiskData';

export const DISTRICT_RISK_DATA = _CANONICAL.map((d) => ({
  ...d,
  // Legacy field aliases used by other pages
  name: d.district,
  cloudburstProb: d.cloudburstProbability,
  flashFloodProb: d.flashFloodProbability,
  vulnerability: (() => {
    switch (d.riskLevel) {
      case 'EXTREME': return 'Critical';
      case 'VERY HIGH': return 'Critical';
      case 'HIGH': return 'High';
      case 'MODERATE': return 'Moderate';
      default: return 'Low';
    }
  })(),
  populationExposed: '—',
}));
