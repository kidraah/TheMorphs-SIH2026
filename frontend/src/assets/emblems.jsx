import React from 'react';

/**
 * High-fidelity government insignia components rendered in SVG
 * for clean, razor-sharp display at all zoom levels.
 */

export const AshokaEmblem = ({ className = "h-14 w-auto" }) => (
  <svg viewBox="0 0 100 125" className={className} fill="currentColor">
    {/* Ashoka Lion Capital Emblem representation */}
    <g fill="#1E293B">
      {/* Central Lion Head */}
      <path d="M50,15 C44,15 40,20 40,26 C40,30 42,33 45,35 C42,37 40,40 40,44 C40,50 44,55 50,55 C56,55 60,50 60,44 C60,40 58,37 55,35 C58,33 60,30 60,26 C60,20 56,15 50,15 Z" />
      {/* Left Lion Head */}
      <path d="M35,22 C30,22 26,26 26,31 C26,35 28,38 31,40 C28,42 26,45 26,49 C26,54 30,58 35,58 C38,58 40,56 42,53 C40,48 40,43 41,38 C38,36 36,33 35,28 Z" opacity="0.9" />
      {/* Right Lion Head */}
      <path d="M65,22 C70,22 74,26 74,31 C74,35 72,38 69,40 C72,42 74,45 74,49 C74,54 70,58 65,58 C62,58 60,56 58,53 C60,48 60,43 59,38 C62,36 64,33 65,28 Z" opacity="0.9" />
      {/* Base Abacus */}
      <rect x="20" y="60" width="60" height="7" rx="1.5" />
      {/* Ashoka Chakra Wheel */}
      <circle cx="50" cy="74" r="6.5" fill="none" stroke="#1E293B" strokeWidth="1.8" />
      <circle cx="50" cy="74" r="1.5" fill="#1E293B" />
      {/* Bull and Horse motifs on abacus */}
      <path d="M30,73 C30,71 33,71 35,74 C34,76 31,76 30,73 Z" />
      <path d="M66,73 C66,71 69,71 71,74 C70,76 67,76 66,73 Z" />
      {/* Lotus Bell Base */}
      <path d="M22,81 C30,80 70,80 78,81 C75,90 65,94 50,94 C35,94 25,90 22,81 Z" />
      {/* Satyameva Jayate Banner */}
      <rect x="25" y="98" width="50" height="6" rx="1" fill="#334155" />
      <text x="50" y="103" textAnchor="middle" fontSize="4.5" fill="#FFFFFF" fontWeight="bold" letterSpacing="0.5">सत्यमेव जयते</text>
    </g>
  </svg>
);

export const ImdLogo = ({ className = "h-14 w-auto" }) => (
  <svg viewBox="0 0 100 100" className={className}>
    {/* Outer border & Seal */}
    <circle cx="50" cy="50" r="46" fill="#0A3871" />
    <circle cx="50" cy="50" r="42" fill="#FFFFFF" stroke="#0A3871" strokeWidth="2" />
    {/* Inner Globe */}
    <circle cx="50" cy="50" r="32" fill="#E0F2FE" stroke="#0284C7" strokeWidth="1.5" />
    <path d="M25,50 A25,25 0 0,0 75,50 A25,12 0 0,0 25,50" fill="none" stroke="#38BDF8" strokeWidth="1" />
    <path d="M25,50 A25,25 0 0,1 75,50 A25,12 0 0,1 25,50" fill="none" stroke="#38BDF8" strokeWidth="1" />
    <line x1="50" y1="18" x2="50" y2="82" stroke="#38BDF8" strokeWidth="1" />
    
    {/* Clouds & Sun / Lightning */}
    <circle cx="43" cy="40" r="7" fill="#F59E0B" opacity="0.9" />
    <path d="M38,48 C36,44 40,40 45,42 C47,38 54,39 56,43 C60,43 62,47 60,50 C58,53 38,53 38,48 Z" fill="#94A3B8" />
    {/* Lightning Bolt */}
    <polygon points="52,44 46,54 50,54 47,63 56,51 51,51" fill="#E11D48" />
    
    {/* Sanskrit Motto Arc */}
    <path id="curveTop" d="M 20,50 A 30,30 0 0,1 80,50" fill="none" />
    <path id="curveBottom" d="M 80,50 A 30,30 0 0,1 20,50" fill="none" />
    <text fontSize="4.8" fontWeight="bold" fill="#0A3871">
      <textPath href="#curveTop" startOffset="50%" textAnchor="middle">
        आदित्यात् जायते वृष्टिः
      </textPath>
    </text>
    <text fontSize="4" fontWeight="bold" fill="#0A3871">
      <textPath href="#curveBottom" startOffset="50%" textAnchor="middle">
        IMD • 1875
      </textPath>
    </text>
  </svg>
);

export const MoesLogo = ({ className = "h-14 w-auto" }) => (
  <svg viewBox="0 0 160 80" className={className}>
    <g fill="#0A3871">
      {/* Ashoka Top motif */}
      <path d="M25,15 C22,15 20,18 20,22 C20,25 22,27 25,29 C22,31 20,34 20,38 C20,44 24,47 28,47 C32,47 36,44 36,38 C36,34 34,31 31,29 C34,27 36,25 36,22 C36,18 34,15 25,15 Z" fill="#334155" />
      <rect x="14" y="49" width="28" height="4" fill="#334155" rx="1" />
      <path d="M16,55 C20,54 36,54 40,55 C38,60 34,63 28,63 C22,63 18,60 16,55 Z" fill="#334155" />
      
      {/* MoES Text Header */}
      <text x="50" y="32" fontSize="13" fontWeight="bold" fill="#1E293B">Ministry of</text>
      <text x="50" y="48" fontSize="13" fontWeight="bold" fill="#0A3871">Earth</text>
      <text x="50" y="64" fontSize="13" fontWeight="bold" fill="#0A3871">Sciences</text>
    </g>
  </svg>
);

export const Imd150Logo = ({ className = "h-14 w-auto" }) => (
  <svg viewBox="0 0 140 80" className={className}>
    {/* 150 Years of Service Logo */}
    <g>
      {/* Bold 150 Number with tricolor/blue touch */}
      <text x="5" y="46" fontSize="40" fontWeight="900" fill="#EA580C" fontFamily="sans-serif">1</text>
      <text x="26" y="46" fontSize="40" fontWeight="900" fill="#0A3871" fontFamily="sans-serif">5</text>
      <text x="52" y="46" fontSize="40" fontWeight="900" fill="#16A34A" fontFamily="sans-serif">0</text>
      
      <circle cx="95" cy="30" r="14" fill="#E0F2FE" stroke="#0A3871" strokeWidth="1.5" />
      <polygon points="95,20 90,30 94,30 91,40 99,28 95,28" fill="#DC2626" />
      
      <text x="8" y="59" fontSize="6.5" fontWeight="bold" fill="#334155">1875 - 2025</text>
      <text x="8" y="69" fontSize="6" fontWeight="bold" fill="#0A3871">150 YEARS OF SERVICE</text>
      <text x="8" y="76" fontSize="5" fill="#64748B">TO THE NATION</text>
    </g>
  </svg>
);
