import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from '../components/Header';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import Footer from '../components/Footer';
import AlertDetailModal from '../components/AlertDetailModal';

export const MainLayout = () => {
  return (
    <div className="min-h-screen bg-[#F4F6F9] flex flex-col">
      {/* Top Government Header */}
      <Header />

      {/* Navy Blue Nav + Sub-Header Title Banner */}
      <Navbar />

      {/* Portal Body: Sidebar + Main Content Column */}
      <div className="flex flex-1 overflow-hidden border-x border-slate-200 max-w-[1720px] w-full mx-auto bg-[#F4F6F9]">
        {/* Fixed-width Left Sidebar */}
        <Sidebar />

        {/* Scrollable Main Content Area */}
        <main className="flex-1 overflow-y-auto p-3 sm:p-3.5 min-w-0">
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
