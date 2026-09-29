import React from 'react';
import { ShieldAlert, MapPin, Database, Activity, Info, Calendar, Radio } from 'lucide-react';

export const Limitations: React.FC = () => {
  return (
    <section className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 shadow-xs">
      <div className="flex items-center justify-between pb-3 border-b border-slate-200 mb-4">
        <div>
          <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-amber-600" />
            Scope & Limitations
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Prototype scope, reference targets, and validation boundaries.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
        {/* Item 1: Demonstration Dataset Coverage */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
          <div className="font-bold text-slate-800 mb-1.5 flex items-center gap-1.5">
            <Calendar className="w-4 h-4 text-sky-700" />
            Temporal Scope
          </div>
          <p className="text-slate-600 leading-relaxed">
            The demonstration dataset covers <strong>January–September 2020</strong> (274 daily ocean snapshots).
          </p>
        </div>

        {/* Item 2: GLORYS Reanalysis Reference */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
          <div className="font-bold text-slate-800 mb-1.5 flex items-center gap-1.5">
            <Database className="w-4 h-4 text-sky-700" />
            Reanalysis Training Target
          </div>
          <p className="text-slate-600 leading-relaxed">
            GLORYS is used as the reanalysis training/reference target, <strong>not observational ground truth</strong>.
          </p>
        </div>

        {/* Item 3: ARGO Independent Observational Check */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
          <div className="font-bold text-slate-800 mb-1.5 flex items-center gap-1.5">
            <Activity className="w-4 h-4 text-sky-700" />
            ARGO Observational Check
          </div>
          <p className="text-slate-600 leading-relaxed">
            ARGO provides an independent observational check using <strong>selected September 2020 snapshots</strong> (497 matched profile-depth observations across 28 floats).
          </p>
        </div>

        {/* Item 4: Prototype Nature */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
          <div className="font-bold text-slate-800 mb-1.5 flex items-center gap-1.5">
            <Radio className="w-4 h-4 text-sky-700" />
            Service Status
          </div>
          <p className="text-slate-600 leading-relaxed">
            The current prototype is a research prototype, <strong>not a live operational satellite service</strong>.
          </p>
        </div>

        {/* Item 5: Domain & Depth Limits */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
          <div className="font-bold text-slate-800 mb-1.5 flex items-center gap-1.5">
            <MapPin className="w-4 h-4 text-sky-700" />
            Domain & Vertical Range
          </div>
          <p className="text-slate-600 leading-relaxed">
            Predictions are strictly limited to the <strong>North Indian Ocean domain (5°N–30°N, 45°E–105°E)</strong> and 15 standard depths down to <strong>1000m</strong>.
          </p>
        </div>

        {/* Item 6: Generalization Scope */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
          <div className="font-bold text-slate-800 mb-1.5 flex items-center gap-1.5">
            <Info className="w-4 h-4 text-sky-700" />
            Generalization Scope
          </div>
          <p className="text-slate-600 leading-relaxed">
            Performance outside the evaluated period and region requires further observational validation.
          </p>
        </div>
      </div>
    </section>
  );
};
