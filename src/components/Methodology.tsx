import React from 'react';
import { Cpu, CheckCircle2, Sliders, Zap, Layers } from 'lucide-react';

export const Methodology: React.FC = () => {
  return (
    <section className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 mb-6 gap-2">
        <div>
          <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Cpu className="w-5 h-5 text-sky-700" />
            OceanEmbed Architecture & Reconstruction Pipeline
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Deep learning framework mapping surface satellite observation grids to vertical subsurface ocean temperature.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono bg-sky-50 text-sky-800 px-2.5 py-1 rounded border border-sky-200 font-medium">
            OceanEmbed v3
          </span>
        </div>
      </div>

      {/* 4-Stage Architecture Pipeline */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-6">
        {/* Stage 1: Surface Observation Inputs */}
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 mb-2">
              <span className="font-bold text-sky-700">STAGE 01</span>
              <span>14 Channels</span>
            </div>
            <h3 className="text-sm font-bold text-slate-800 mb-2">
              Surface Observation Inputs
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed mb-3">
              Daily 0.25° × 0.25° gridded fields combining 7 surface variables alongside 7 binary quality masks.
            </p>
            <div className="space-y-1 text-[11px] font-mono text-slate-600 bg-white p-2.5 rounded border border-slate-200">
              <div>• Sea Surface Temperature (SST)</div>
              <div>• Sea Surface Salinity (SSS)</div>
              <div>• Sea Surface Height (SSH/SLA)</div>
              <div>• Surface U/V Currents</div>
              <div>• 10m U/V Atmospheric Winds</div>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-200 text-[10px] font-mono text-slate-500">
            7 Variables + 7 Validity Masks
          </div>
        </div>

        {/* Stage 2: Multi-Scale Spatial Encoder */}
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 mb-2">
              <span className="font-bold text-sky-700">STAGE 02</span>
              <span>U-Net Encoder</span>
            </div>
            <h3 className="text-sm font-bold text-slate-800 mb-2">
              Multi-scale Spatial Encoder
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed mb-3">
              Learns spatial and cross-variable surface patterns across hierarchical resolutions.
            </p>
            <div className="space-y-1 text-[11px] font-mono text-slate-600 bg-white p-2.5 rounded border border-slate-200">
              <div>• DoubleConv blocks (32 → 64 → 128)</div>
              <div>• 3×3 Convolutions + BatchNorm + ReLU</div>
              <div>• MaxPool2d spatial downsampling</div>
              <div>• Lateral feature skip connections</div>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-200 text-[10px] font-mono text-slate-500">
            Input: [14, 101, 241] Grid
          </div>
        </div>

        {/* Stage 3: Ocean Latent Embedding */}
        <div className="p-4 bg-sky-50/50 border border-sky-200 rounded-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-[11px] font-mono text-sky-800 mb-2">
              <span className="font-bold text-sky-800">STAGE 03</span>
              <span>128-d Embedding</span>
            </div>
            <h3 className="text-sm font-bold text-sky-950 mb-2">
              Ocean Latent Embedding
            </h3>
            <p className="text-xs text-sky-900 leading-relaxed mb-3">
              Learns a compact representation of spatial and cross-variable surface-state patterns at the network bottleneck.
            </p>
            <div className="space-y-1 text-[11px] font-mono text-sky-900 bg-white/90 p-2.5 rounded border border-sky-200">
              <div>• Bottleneck dimension: 128 channels</div>
              <div>• Compact surface-state encoding</div>
              <div>• Spatial decoder with skip routing</div>
              <div>• Transpose-conv spatial upsampling</div>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-sky-200 text-[10px] font-mono text-sky-700">
            Bottleneck: [128, 12, 30]
          </div>
        </div>

        {/* Stage 4: Parallel Multi-Depth Projection Head */}
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 mb-2">
              <span className="font-bold text-sky-700">STAGE 04</span>
              <span>15 Depths Head</span>
            </div>
            <h3 className="text-sm font-bold text-slate-800 mb-2">
              Parallel Multi-depth Projection Head
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed mb-3">
              Maps the learned representation to temperature at 15 standard depths simultaneously using a 1×1 convolution.
            </p>
            <div className="space-y-1 text-[11px] font-mono text-slate-600 bg-white p-2.5 rounded border border-slate-200">
              <div>• Surface: 0, 5, 10, 20, 30, 50m</div>
              <div>• Thermocline: 75, 100, 125, 150m</div>
              <div>• Intermediate: 200, 300m</div>
              <div>• Deep: 500, 700, 1000m</div>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-200 text-[10px] font-mono text-slate-500 flex items-center justify-between">
            <span>Output: [15, 101, 241] Field</span>
            <span>1.28M Parameters</span>
          </div>
        </div>
      </div>

      {/* Architecture Speedup & Temporal Split Panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        {/* Panel 1: Parallel Projection Engineering Speedup */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
          <div className="font-bold text-slate-800 mb-1.5 flex items-center gap-1.5">
            <Zap className="w-4 h-4 text-sky-700" />
            Parallel Multi-Depth Architecture Speedup (7.2×)
          </div>
          <p className="text-slate-600 leading-relaxed">
            Replacing the sequential single-channel shared decoder loop with a parallel multi-depth projection head accelerated epoch training time from approximately 172 seconds to approximately 24 seconds per epoch under identical conditions.
          </p>
        </div>

        {/* Panel 2: Strict Temporal Splitting */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
          <div className="font-bold text-slate-800 mb-1.5 flex items-center gap-1.5">
            <CheckCircle2 className="w-4 h-4 text-sky-700" />
            Temporal Split (Future-Period Leakage Prevention)
          </div>
          <p className="text-slate-600 leading-relaxed mb-2">
            The model was trained on January–July 2020 data, selected using August 2020 validation, and evaluated on a completely held-out September 2020 period.
          </p>
          <div className="flex items-center gap-2 font-mono text-[10px] text-slate-600 bg-white px-2 py-1 rounded border border-slate-200">
            <span className="font-semibold text-slate-800">TRAIN:</span> Jan–Jul 2020
            <span className="text-slate-300">|</span>
            <span className="font-semibold text-slate-800">VAL:</span> Aug 2020
            <span className="text-slate-300">|</span>
            <span className="font-semibold text-sky-800">TEST:</span> Sep 2020
          </div>
        </div>
      </div>
    </section>
  );
};
