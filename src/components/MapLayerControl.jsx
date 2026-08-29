import React, { useState } from 'react';
import { Layers, ChevronDown, Check } from 'lucide-react';
import { BASEMAPS, MAP_OVERLAYS } from '../data/mapLayers';
import { useAppStore } from '../store/useAppStore';

export const MapLayerControl = () => {
  const {
    basemap, setBasemap,
    mapLayers, toggleMapLayer,
    showBorders, toggleBorders,
  } = useAppStore();

  const [isOpen, setIsOpen] = useState(false);

  const activeBasemapObj = BASEMAPS.find((b) => b.id === basemap) || BASEMAPS[0];

  return (
    <div className="absolute top-3 right-3 z-[500] font-sans">
      {/* Compact Header Button */}
      <div className="bg-white/95 border border-[#E2E2E2] rounded shadow-md text-xs">
        <button
          type="button"
          onClick={() => setIsOpen((prev) => !prev)}
          className="flex items-center justify-between gap-2 px-2.5 py-1.5 font-bold text-[#172033] hover:bg-[#FFF7EA] transition-colors w-full"
        >
          <div className="flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-[#E87516]" />
            <span className="uppercase text-[10.5px] tracking-wider text-slate-600">Map View:</span>
            <span className="text-[11px] text-[#C85D00] font-extrabold">{activeBasemapObj.name}</span>
          </div>
          <ChevronDown className={`w-3.5 h-3.5 text-slate-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </button>

        {/* Panel Menu */}
        {isOpen && (
          <div className="border-t border-[#E2E2E2] p-2.5 bg-white w-60 shadow-xl rounded-b text-xs space-y-2.5">
            {/* BASEMAPS SECTION */}
            <div>
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wide border-b border-slate-100 pb-1 mb-1.5">
                Base Maps
              </div>
              <div className="space-y-1">
                {BASEMAPS.map((b) => (
                  <label
                    key={b.id}
                    className={`flex items-center justify-between p-1.5 rounded cursor-pointer transition-colors ${
                      basemap === b.id
                        ? 'bg-[#FFF1DD] text-[#C85D00] font-bold border border-orange-200'
                        : 'text-[#172033] hover:bg-[#FFF7EA] border border-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <input
                        type="radio"
                        name="basemap-option"
                        value={b.id}
                        checked={basemap === b.id}
                        onChange={() => setBasemap(b.id)}
                        className="accent-[#E87516]"
                      />
                      <span className="text-[11px]">{b.name}</span>
                    </div>
                    {basemap === b.id && <Check className="w-3 h-3 text-[#E87516]" />}
                  </label>
                ))}
              </div>
            </div>

            {/* OVERLAYS SECTION */}
            <div>
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wide border-b border-slate-100 pb-1 mb-1.5">
                Data Overlays
              </div>
              <div className="space-y-1">
                {MAP_OVERLAYS.map((overlay) => (
                  <label
                    key={overlay.key}
                    className="flex items-center justify-between p-1.5 rounded hover:bg-[#FFF7EA] cursor-pointer text-[#172033] transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={!!mapLayers[overlay.key]}
                        onChange={() => toggleMapLayer(overlay.key)}
                        className="rounded accent-[#E87516]"
                      />
                      <span className="text-[11px]">{overlay.name}</span>
                    </div>
                  </label>
                ))}
                <label className="flex items-center justify-between p-1.5 rounded hover:bg-[#FFF7EA] cursor-pointer text-[#172033] transition-colors">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={showBorders}
                      onChange={toggleBorders}
                      className="rounded accent-[#E87516]"
                    />
                    <span className="text-[11px]">District Boundaries</span>
                  </div>
                </label>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MapLayerControl;
