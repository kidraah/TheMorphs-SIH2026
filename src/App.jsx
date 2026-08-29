import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import Dashboard from './pages/Dashboard';
import LiveMapPage from './pages/LiveMapPage';
import RiskOutlookPage from './pages/RiskOutlookPage';
import AlertsPage from './pages/AlertsPage';
import ForecastPage from './pages/ForecastPage';
import DataLayersPage from './pages/DataLayersPage';
import ReportsPage from './pages/ReportsPage';
import SettingsPage from './pages/SettingsPage';
import AboutSystemPage from './pages/AboutSystemPage';

export function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="live-map" element={<LiveMapPage />} />
          <Route path="risk-outlook" element={<RiskOutlookPage />} />
          <Route path="alerts" element={<AlertsPage />} />
          <Route path="forecast" element={<ForecastPage />} />
          <Route path="data-layers" element={<DataLayersPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="about-system" element={<AboutSystemPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
