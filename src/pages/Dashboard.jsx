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
    <div className="space-y-3.5 max-w-[1720px] mx-auto">
      {/* 1. Full-Width High-Risk Alert Banner */}
      <AlertBanner />

      {/* 2. Main Dashboard Top Section: 3-Column Layout
          Col 1 (~58%): Live Risk Map (Expanded width)
          Col 2 (~25%): Risk Timeline
          Col 3 (~17%): Vertical Risk Status Cards (Slim compact width) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5 min-h-[540px]">
        
        {/* COLUMN 1: Live Risk Map (58% on desktop) */}
        <div className="lg:col-span-7 xl:col-span-7 flex flex-col h-[520px] lg:h-[540px] xl:h-[560px]">
          <RiskMap height="h-full" />
        </div>

        {/* COLUMN 2: Risk Timeline (25% on desktop) */}
        <div className="lg:col-span-3 xl:col-span-3 flex flex-col h-[520px] lg:h-[540px] xl:h-[560px]">
          <RiskTimeline />
        </div>

        {/* COLUMN 3: Vertical Risk Status Cards (Slim ~17% on desktop) */}
        <div className="lg:col-span-2 xl:col-span-2 flex flex-col h-auto lg:h-[540px] xl:h-[560px]">
          <RiskCards layout="vertical" />
        </div>
      </div>

      {/* 3. Below Main Dashboard: Supporting Analytics & Operations Panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3.5 min-h-[220px]">
        <ExplainableAI triggers={mockXaiTriggers} />
        <RecentAlerts alerts={mockRecentAlerts} />
        <PrecipitationNowcast />
        <DataSources sources={mockDataSources} />
        <SystemPerformance metrics={mockSystemPerformance} />
      </div>
    </div>
  );
};

export default Dashboard;
