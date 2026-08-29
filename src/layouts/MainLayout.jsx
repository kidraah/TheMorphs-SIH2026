import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from '../components/Header';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import AlertDetailModal from '../components/AlertDetailModal';

export const MainLayout = () => {
  return (
    <div className="min-h-screen bg-[#FFFFFF] flex flex-col">
      {/* Top Government Header */}
      <Header />

      {/* Saffron Primary Top Navigation + Sub-Header Title Banner */}
      <Navbar />

      {/* Portal Body: Full-Width Main Content Column */}
      <div className="flex flex-1 overflow-hidden max-w-[1720px] w-full mx-auto bg-[#FFFFFF]">
        {/* Scrollable Main Content Area - Full Width */}
        <main className="flex-1 overflow-y-auto p-3 sm:p-4 min-w-0 bg-[#FAFAFA]">
          <Outlet />
        </main>
      </div>

      {/* Institutional Footer */}
      <Footer />

      {/* Global Alert Detail Modal */}
      <AlertDetailModal />
    </div>
  );
};

export default MainLayout;
