import React from 'react';
import { Thermometer, Waves, Flame, ArrowDown, Activity } from 'lucide-react';
import { PredictionResponse } from '../types';

interface ProfileMetricsProps {
  prediction: PredictionResponse | null;
}

export const ProfileMetrics: React.FC<ProfileMetricsProps> = ({ prediction }) => {
  if (!prediction) {
    return null;
  }

  const { depths_m, temperatures_c, oceanographic_indicators, inference_ms, total_latency_ms } =
    prediction;

  // Extract key benchmark levels
  const surfaceT = temperatures_c[0];
  const idx100 = depths_m.indexOf(100);
  const temp100 = idx100 !== -1 ? temperatures_c[idx100] : null;

  const idx500 = depths_m.indexOf(500);
  const temp500 = idx500 !== -1 ? temperatures_c[idx500] : null;

  const idx1000 = depths_m.indexOf(1000);
  const temp1000 = idx1000 !== -1 ? temperatures_c[idx1000] : null;

  const mld = oceanographic_indicators?.mixed_layer_depth_m;
  const thermocline = oceanographic_indicators?.thermocline_depth_m;
  const ohc300 = oceanographic_indicators?.ocean_heat_content_300m_gj_m2;

  return (
    <div className="space-y-4">
      {/* 1. Key Vertical Strata Temperatures */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs">
        <div className="flex items-center justify-between pb-2 border-b border-slate-100 mb-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
            <Thermometer className="w-3.5 h-3.5 text-sky-700" />
            Key Subsurface Horizons
          </h3>
          <span className="text-[10px] font-mono text-slate-400">Reconstructed</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
            <span className="text-[10px] font-mono text-slate-500 uppercase block">Surface (0m)</span>
            <span className="text-lg font-bold font-mono text-slate-900">
              {surfaceT.toFixed(2)}
              <span className="text-xs font-normal text-slate-500 ml-0.5">°C</span>
            </span>
          </div>

          <div className="p-2.5 bg-amber-50/50 border border-amber-200 rounded-lg">
            <span className="text-[10px] font-mono text-amber-800 uppercase block">
              Thermocline (100m)
            </span>
            <span className="text-lg font-bold font-mono text-amber-950">
              {temp100 !== null ? temp100.toFixed(2) : '—'}
              <span className="text-xs font-normal text-amber-700 ml-0.5">°C</span>
            </span>
          </div>

          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
            <span className="text-[10px] font-mono text-slate-500 uppercase block">Deep (500m)</span>
            <span className="text-lg font-bold font-mono text-slate-900">
              {temp500 !== null ? temp500.toFixed(2) : '—'}
              <span className="text-xs font-normal text-slate-500 ml-0.5">°C</span>
            </span>
          </div>

          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
            <span className="text-[10px] font-mono text-slate-500 uppercase block">Deep (1000m)</span>
            <span className="text-lg font-bold font-mono text-slate-900">
              {temp1000 !== null ? temp1000.toFixed(2) : '—'}
              <span className="text-xs font-normal text-slate-500 ml-0.5">°C</span>
            </span>
          </div>
        </div>
      </div>

      {/* 2. Derived Physical Oceanographic Indicators */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs">
        <div className="flex items-center justify-between pb-2 border-b border-slate-100 mb-3">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
              <Waves className="w-3.5 h-3.5 text-teal-700" />
              Profile-Derived Indicators
            </h3>
            <span className="text-[10px] text-slate-500 font-medium">
              Calculated from reconstructed vertical temperature profile
            </span>
          </div>
          <span className="text-[10px] font-mono text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
            Profile-Derived
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* Mixed Layer Depth */}
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="flex items-center justify-between text-xs text-slate-600 mb-1">
              <span className="font-semibold">Mixed Layer Depth (MLD)</span>
            </div>
            <div className="text-lg font-bold font-mono text-slate-900">
              {mld !== null ? `${mld.toFixed(1)} m` : '—'}
            </div>
            <span className="text-[10px] text-slate-500 block mt-0.5">
              ΔT = 0.2°C surface threshold
            </span>
          </div>

          {/* Thermocline Depth */}
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="flex items-center justify-between text-xs text-slate-600 mb-1">
              <span className="font-semibold">Thermocline Depth</span>
            </div>
            <div className="text-lg font-bold font-mono text-slate-900">
              {thermocline !== null ? `${thermocline.toFixed(1)} m` : '—'}
            </div>
            <span className="text-[10px] text-slate-500 block mt-0.5">
              Peak vertical gradient |dT/dz|
            </span>
          </div>

          {/* Ocean Heat Content 300m */}
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="flex items-center justify-between text-xs text-slate-600 mb-1">
              <span className="font-semibold">Heat Content (OHC300)</span>
            </div>
            <div className="text-lg font-bold font-mono text-slate-900">
              {ohc300 !== null ? `${ohc300.toFixed(2)} GJ/m²` : '—'}
            </div>
            <span className="text-[10px] text-slate-500 block mt-0.5">
              Integrated upper 300 meters
            </span>
          </div>
        </div>
      </div>

      {/* 3. Real Inference Telemetry Strip */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5 flex items-center justify-between text-xs font-mono text-slate-600">
        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-sky-700" />
          <span>Model: <strong className="text-slate-800">{prediction.model}</strong></span>
          <span className="text-slate-300">•</span>
          <span>Status: <strong className="text-emerald-700">Completed</strong></span>
        </div>
        <div className="flex items-center gap-2">
          <span>Inference: <strong className="text-slate-800">{inference_ms} ms</strong></span>
          <span className="text-slate-300">•</span>
          <span>Roundtrip: <strong className="text-slate-800">{total_latency_ms} ms</strong></span>
        </div>
      </div>
    </div>
  );
};
