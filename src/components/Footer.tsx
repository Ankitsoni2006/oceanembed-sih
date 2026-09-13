import React from 'react';
import { Layers, ShieldCheck, ExternalLink, GitBranch, Cpu } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-200 bg-slate-50 text-slate-600 mt-12 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pb-6 border-b border-slate-200 text-xs">
          {/* Col 1: Project Identity */}
          <div>
            <div className="flex items-center gap-2 mb-2 font-bold text-slate-800">
              <Layers className="w-4 h-4 text-sky-700" />
              <span>OceanEmbed</span>
            </div>
            <p className="text-slate-500 leading-relaxed">
              Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations.
            </p>
            <div className="mt-2 text-[11px] text-slate-400 font-mono">
              North Indian Ocean Domain (5°N–30°N, 45°E–105°E)
            </div>
          </div>

          {/* Col 2: Specifications */}
          <div>
            <div className="font-bold text-slate-800 mb-2 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-sky-700" />
              <span>Model & Domain Specifications</span>
            </div>
            <ul className="space-y-1 font-mono text-[11px] text-slate-500">
              <li>• Architecture: OceanEmbed v3</li>
              <li>• Parameters: 1,275,934 weights</li>
              <li>• Target Domain: 5°N–30°N, 45°E–105°E (0.25° grid)</li>
              <li>• Standard Depths: 15 levels (0m to 1000m)</li>
              <li>• Temporal Cadence: Daily snapshots</li>
            </ul>
          </div>

          {/* Col 3: Validation Reference */}
          <div>
            <div className="font-bold text-slate-800 mb-2 flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-700" />
              <span>Validation Reference</span>
            </div>
            <ul className="space-y-1 text-slate-500">
              <li>• GLORYS Reanalysis Reference: <strong>0.8601°C RMSE</strong></li>
              <li>• ARGO Observational Check: <strong>0.7973°C RMSE</strong></li>
              <li>• Evaluation Sample: 497 matched profile-depth observations</li>
              <li>• Evaluation Split: Held-out September 2020</li>
            </ul>
          </div>
        </div>

        {/* Bottom attribution */}
        <div className="pt-4 flex flex-col sm:flex-row items-center justify-between text-[11px] text-slate-500 gap-2">
          <div>
            OceanEmbed • Subsurface Ocean Analysis
          </div>
          <div className="flex items-center gap-4">
            <span className="font-mono">Research Prototype</span>
            <span className="font-mono">Inference Engine: PyTorch (CPU/CUDA)</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
