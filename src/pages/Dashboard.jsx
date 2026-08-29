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

      {/* 3. Middle Row: Live Risk Map (Left) + Timeline & XAI (Right) */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3" style={{ minHeight: 520 }}>
        {/* LEFT: Live Risk Map – Expanded full-width viewport */}
        <div className="xl:col-span-8 flex flex-col" style={{ minHeight: 520 }}>
          <RiskMap height="h-full" />
        </div>

        {/* RIGHT: Timeline stacked above Explainable AI */}
        <div className="xl:col-span-4 flex flex-col gap-3" style={{ minHeight: 520 }}>
          {/* Risk Timeline Chart */}
          <div style={{ height: 245 }}>
            <RiskTimeline data={mockRiskTimeline} />
          </div>
          {/* Explainable AI Key Triggers */}
          <div className="flex-1">
            <ExplainableAI triggers={mockXaiTriggers} />
          </div>
        </div>
      </div>

      {/* 4. Bottom Row: Four Operational Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3" style={{ minHeight: 210 }}>
        <RecentAlerts alerts={mockRecentAlerts} />
        <PrecipitationNowcast />
        <DataSources sources={mockDataSources} />
        <SystemPerformance metrics={mockSystemPerformance} />
      </div>
    </div>
  );
};

export default Dashboard;
