import React from 'react';
import { Compass, Grid3X3, Layers, Calendar } from 'lucide-react';

export const Hero: React.FC = () => {
  return (
    <section className="bg-slate-50 border-b border-slate-200 py-8 sm:py-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-sky-100 text-sky-800 text-xs font-semibold mb-3 border border-sky-200">
            <span>OCEAN STATE RECONSTRUCTION</span>
          </div>

          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight">
            Reconstruct the Ocean Below the Surface
          </h1>

          <p className="mt-2.5 text-base sm:text-lg text-slate-600 leading-relaxed">
            Estimate vertical temperature profiles from the surface down to 1000&nbsp;m using seven satellite-derived surface observations across the North Indian Ocean.
          </p>
        </div>

        {/* Technical Specification Strip */}
        <div className="mt-6 pt-5 border-t border-slate-200 grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
          <div className="flex items-center gap-3 bg-white p-3 rounded-lg border border-slate-200 shadow-xs">
            <div className="w-8 h-8 rounded bg-slate-100 text-slate-700 flex items-center justify-center shrink-0">
              <Compass className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Domain</div>
              <div className="text-xs font-bold text-slate-800">5°N–30°N, 45°E–105°E</div>
            </div>
          </div>

          <div className="flex items-center gap-3 bg-white p-3 rounded-lg border border-slate-200 shadow-xs">
            <div className="w-8 h-8 rounded bg-slate-100 text-slate-700 flex items-center justify-center shrink-0">
              <Grid3X3 className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Spatial Grid</div>
              <div className="text-xs font-bold text-slate-800">0.25° × 0.25° (101×241)</div>
            </div>
          </div>

          <div className="flex items-center gap-3 bg-white p-3 rounded-lg border border-slate-200 shadow-xs">
            <div className="w-8 h-8 rounded bg-slate-100 text-slate-700 flex items-center justify-center shrink-0">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Vertical Column</div>
              <div className="text-xs font-bold text-slate-800">15 Depths (0–1000m)</div>
            </div>
          </div>

          <div className="flex items-center gap-3 bg-white p-3 rounded-lg border border-slate-200 shadow-xs">
            <div className="w-8 h-8 rounded bg-slate-100 text-slate-700 flex items-center justify-center shrink-0">
              <Calendar className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Cadence</div>
              <div className="text-xs font-bold text-slate-800">Daily Snapshots (2020)</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
