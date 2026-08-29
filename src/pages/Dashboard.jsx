import React from 'react';
import AlertBanner from '../components/AlertBanner';
import RiskCards from '../components/RiskCards';
import RiskMap from '../components/RiskMap';
import RiskTimeline from '../components/RiskTimeline';
import ExplainableAI from '../components/ExplainableAI';
import RecentAlerts from '../components/RecentAlerts';
import PrecipitationNowcast from '../components/PrecipitationNowcast';
import DataSources from '../components/DataSources';
import SystemPerformance from '../components/SystemPerformance';
import {
  mockRiskSummary,
  mockRiskTimeline,
  mockXaiTriggers,
  mockRecentAlerts,
  mockDataSources,
  mockSystemPerformance,
} from '../data/mockData';

export const Dashboard = () => {
  return (
    <div className="space-y-3">
      {/* 1. High-Risk Alert Banner */}
      <AlertBanner />

      {/* 2. Five KPI Risk Summary Cards */}
      <RiskCards riskSummary={mockRiskSummary} />

      {/* 3. Middle Row: Live Risk Map (Left ~62%) + Timeline & XAI (Right ~38%) */}
      <div className="grid gap-3" style={{ gridTemplateColumns: '1fr 340px' }}>
        {/* LEFT: Live Risk Map – taller height to dominate this row */}
        <div style={{ minHeight: 480 }}>
          <RiskMap height="h-full" />
        </div>

        {/* RIGHT: Timeline stacked above Explainable AI */}
        <div className="flex flex-col gap-3" style={{ minHeight: 480 }}>
          {/* Risk Timeline Chart */}
          <div style={{ height: 230 }}>
            <RiskTimeline data={mockRiskTimeline} />
          </div>
          {/* Explainable AI Key Triggers */}
          <div className="flex-1">
            <ExplainableAI triggers={mockXaiTriggers} />
          </div>
        </div>
      </div>

      {/* 4. Bottom Row: Four Equal-Width Operational Metric Cards */}
      <div className="grid grid-cols-4 gap-3" style={{ minHeight: 200 }}>
        <RecentAlerts alerts={mockRecentAlerts} />
        <PrecipitationNowcast />
        <DataSources sources={mockDataSources} />
        <SystemPerformance metrics={mockSystemPerformance} />
      </div>
    </div>
  );
};

export default Dashboard;
