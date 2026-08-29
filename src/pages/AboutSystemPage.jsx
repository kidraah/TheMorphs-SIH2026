import React from 'react';
import { Info, Cpu, ShieldCheck, Database, Award, ExternalLink } from 'lucide-react';
import { AshokaEmblem, ImdLogo, MoesLogo, Imd150Logo } from '../assets/emblems';

export const AboutSystemPage = () => {
  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-white border border-slate-200 rounded p-4 flex flex-wrap items-center justify-between gap-3 shadow-gov-sm">
        <div>
          <span className="text-[10.5px] font-bold text-red-700 uppercase tracking-widest block">
            Smart India Hackathon (SIH 2026) • Problem Statement SIH26077
          </span>
          <h2 className="text-base sm:text-lg font-bold text-[#0A3871] mt-0.5 m-0 flex items-center gap-2">
            <Info className="w-5 h-5 text-blue-700" />
            AI-Driven Hyper-Local Early Warning System (IMD / MoES)
          </h2>
          <p className="text-xs text-slate-600 mt-1 m-0">
            Nowcasting for Severe Thunderstorm, Cloudburst & Flash Flood in Vulnerable Himalayan Terrains
          </p>
        </div>
      </div>

      {/* Grid of details */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Card 1: Mandate & Problem Scope */}
        <div className="bg-white border border-slate-200 rounded p-4 shadow-gov-sm space-y-2.5">
          <div className="flex items-center gap-2 border-b pb-2 text-[#0A3871]">
            <Award className="w-4 h-4 text-blue-700" />
            <h3 className="text-xs font-bold uppercase tracking-wider m-0">Mission & Problem Scope</h3>
          </div>
          <p className="text-xs text-slate-700 leading-relaxed">
            Himalayan cloudbursts and flash floods develop with extreme rapidness (within 1-3 hours) over complex mountainous terrain where conventional numerical models have limited localized skill.
          </p>
          <p className="text-xs text-slate-700 leading-relaxed">
            This platform provides <strong>hyper-local nowcasts (0-6 hours)</strong> with high spatial resolution (~1 km²) and explainable AI insights to empower state disaster management authorities to take proactive evacuation measures.
          </p>
        </div>

        {/* Card 2: AI & Nowcasting Engine */}
        <div className="bg-white border border-slate-200 rounded p-4 shadow-gov-sm space-y-2.5">
          <div className="flex items-center gap-2 border-b pb-2 text-[#0A3871]">
            <Cpu className="w-4 h-4 text-purple-700" />
            <h3 className="text-xs font-bold uppercase tracking-wider m-0">AI Architecture & XAI</h3>
          </div>
          <ul className="text-xs text-slate-700 space-y-1.5 list-disc pl-4 leading-relaxed">
            <li>
              <strong>Spatiotemporal Convective Model:</strong> Multi-scale 3D-CNN and ConvLSTM assimilated on INSAT-3DR rapid-scan infrared and water vapor radiances.
            </li>
            <li>
              <strong>Explainable AI (XAI):</strong> Feature attribution engine ranking thermodynamic stability (CAPE/CIN), cloud top cooling rates, and low-level moisture convergence.
            </li>
            <li>
              <strong>Topographic Routing:</strong> High-resolution DEM hydrological accumulation models for flash flood catchment prediction.
            </li>
          </ul>
        </div>

        {/* Card 3: Institutional Partner Network */}
        <div className="bg-white border border-slate-200 rounded p-4 shadow-gov-sm space-y-2.5">
          <div className="flex items-center gap-2 border-b pb-2 text-[#0A3871]">
            <ShieldCheck className="w-4 h-4 text-emerald-700" />
            <h3 className="text-xs font-bold uppercase tracking-wider m-0">Institutional Collaboration</h3>
          </div>
          <div className="flex flex-wrap items-center gap-3 pt-2">
            <AshokaEmblem className="h-10 w-auto" />
            <ImdLogo className="h-10 w-auto" />
            <MoesLogo className="h-8 w-auto" />
            <Imd150Logo className="h-8 w-auto" />
          </div>
          <p className="text-[11px] text-slate-600 mt-2">
            Deployed under the auspices of the India Meteorological Department (IMD) and Ministry of Earth Sciences (MoES), Government of India.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AboutSystemPage;
