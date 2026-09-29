import React from 'react';
import { Sliders, Target, Thermometer } from 'lucide-react';
import { THERMOCLINE_TOP_M, THERMOCLINE_BOTTOM_M, isInThermoclineBand } from '../lib/ocean';

interface DepthInspectorProps {
  depths: number[];
  temperatures: number[];
  selectedDepth: number | null;
  onSelectDepth: (depth: number) => void;
}

export const DepthInspector: React.FC<DepthInspectorProps> = ({
  depths,
  temperatures,
  selectedDepth,
  onSelectDepth,
}) => {
  // Find current temperature for selected depth
  const activeDepth = selectedDepth !== null ? selectedDepth : depths[0] ?? 0;
  const activeIdx = depths.indexOf(activeDepth);
  const activeTemp = activeIdx !== -1 ? temperatures[activeIdx] : null;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-xs">
      <div className="flex items-center justify-between pb-3 border-b border-slate-200 mb-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
          <Sliders className="w-4 h-4 text-sky-700" />
          Interactive Depth Inspector
        </h3>
        <span className="text-[11px] font-mono text-slate-500">
          15 Standard Levels
        </span>
      </div>

      {/* Selected Depth & Reconstructed Temperature Display */}
      <div className="bg-sky-50/50 border border-sky-200 rounded-lg p-3.5 mb-4 flex items-center justify-between">
        <div>
          <span className="text-[10px] font-mono text-sky-800 uppercase tracking-wider block">
            Selected Vertical Depth
          </span>
          <span className="text-xl font-bold font-mono text-slate-900 flex items-center gap-1.5">
            <Target className="w-4 h-4 text-sky-700" />
            {activeDepth} <span className="text-sm font-normal text-slate-600">meters</span>
          </span>
        </div>

        <div className="text-right">
          <span className="text-[10px] font-mono text-sky-800 uppercase tracking-wider block">
            Reconstructed Temperature
          </span>
          <span className="text-xl font-bold font-mono text-sky-900 flex items-center justify-end gap-1">
            <Thermometer className="w-4 h-4 text-sky-700" />
            {activeTemp !== null ? activeTemp.toFixed(2) : '—'}
            <span className="text-sm font-normal text-slate-600">°C</span>
          </span>
        </div>
      </div>

      {/* Depth Level Selection Pills */}
      <div>
        <div className="text-[11px] font-semibold text-slate-600 mb-2">
          Quick-Select Horizon:
        </div>
        <div className="grid grid-cols-5 sm:grid-cols-8 md:grid-cols-15 gap-1.5">
          {depths.map((d, i) => {
            const isSelected = activeDepth === d;
            const isThermocline = isInThermoclineBand(d);
            const tempVal = temperatures[i];

            return (
              <button
                key={`depth-pill-${d}`}
                type="button"
                onClick={() => onSelectDepth(d)}
                className={`py-1.5 px-1 rounded text-center transition-all border font-mono ${
                  isSelected
                    ? 'bg-sky-800 text-white border-sky-900 font-bold shadow-xs'
                    : isThermocline
                    ? 'bg-amber-50 text-amber-900 border-amber-200 hover:bg-amber-100'
                    : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                }`}
                title={`Depth ${d}m: ${tempVal ? tempVal.toFixed(2) + '°C' : ''}`}
              >
                <div className="text-xs">{d}m</div>
                {tempVal !== undefined && (
                  <div
                    className={`text-[9px] ${
                      isSelected ? 'text-sky-200' : 'text-slate-500'
                    }`}
                  >
                    {tempVal.toFixed(1)}°
                  </div>
                )}
              </button>
            );
          })}
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2.5 text-[10px] text-slate-500">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-xs bg-slate-100 border border-slate-300 inline-block" />
            <span>Surface (0–50m)</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-xs bg-amber-100 border border-amber-300 inline-block" />
            <span>Nominal thermocline ({THERMOCLINE_TOP_M}–{THERMOCLINE_BOTTOM_M}m)</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-xs bg-slate-100 border border-slate-300 inline-block" />
            <span>Intermediate (200–300m)</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-xs bg-slate-100 border border-slate-300 inline-block" />
            <span>Deep (500–1000m)</span>
          </span>
        </div>
      </div>
    </div>
  );
};
