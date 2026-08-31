import React, { useEffect } from 'react';
import AlertBanner from '../components/AlertBanner';
import RiskCards from '../components/RiskCards';
import RiskMap from '../components/RiskMap';
import RiskTimeline from '../components/RiskTimeline';
import ExplainableAI from '../components/ExplainableAI';
import RecentAlerts from '../components/RecentAlerts';
import PrecipitationNowcast from '../components/PrecipitationNowcast';
import DataSources from '../components/DataSources';
import SystemPerformance from '../components/SystemPerformance';
import { useAppStore } from '../store/useAppStore';

export const Dashboard = () => {
  const { 
    fetchDashboardData, 
    isDataLoading, 
    xaiTriggers, 
    alerts, 
    dataSources, 
    systemPerformance 
  } = useAppStore();

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  if (isDataLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen text-slate-500 font-semibold text-lg">
        <span className="animate-pulse">Loading VARUNA AI insights...</span>
      </div>
    );
  }

  return (
    <div className="space-y-3.5 max-w-[1720px] mx-auto">
      {/* 1. Full-Width High-Risk Alert Banner */}
      <AlertBanner />

      {/* 2. Main Dashboard Top Section: 3-Column Layout */}
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
        <ExplainableAI triggers={xaiTriggers} />
        <RecentAlerts alerts={alerts} />
        <PrecipitationNowcast />
        <DataSources sources={dataSources} />
        <SystemPerformance metrics={systemPerformance} />
      </div>
    </div>
  );
};

export default Dashboard;

