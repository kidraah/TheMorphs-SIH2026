import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from '../components/Header';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import Footer from '../components/Footer';
import AlertDetailModal from '../components/AlertDetailModal';

export const MainLayout = () => {
  return (
    <div className="min-h-screen bg-[#FFFFFF] flex flex-col">
      {/* Top Government Header */}
      <Header />

      {/* Saffron Nav + Sub-Header Title Banner */}
      <Navbar />

      {/* Portal Body: Sidebar + Main Content Column */}
      <div className="flex flex-1 overflow-hidden border-x border-[#E2E2E2] max-w-[1720px] w-full mx-auto bg-[#FFFFFF]">
        {/* Fixed-width Left Sidebar */}
        <Sidebar />

        {/* Scrollable Main Content Area */}
        <main className="flex-1 overflow-y-auto p-3 sm:p-3.5 min-w-0 bg-[#FAFAFA]">
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
