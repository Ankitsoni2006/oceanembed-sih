import React, { useState } from 'react';
import { Layers, Info, Thermometer } from 'lucide-react';
import { PredictionResponse, DepthPoint } from '../types';

interface ProfileChartProps {
  prediction: PredictionResponse | null;
  selectedDepth: number | null;
  onSelectDepth: (depth: number) => void;
  isLoading: boolean;
}

// Fixed bounds for oceanographic temperature chart
const T_MIN = 4.0;
const T_MAX = 32.0;

export const ProfileChart: React.FC<ProfileChartProps> = ({
  prediction,
  selectedDepth,
  onSelectDepth,
  isLoading,
}) => {
  const [hoveredPoint, setHoveredPoint] = useState<DepthPoint | null>(null);

  if (!prediction) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs flex flex-col items-center justify-center min-h-[440px] text-center">
        <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 mb-3">
          <Layers className="w-6 h-6" />
        </div>
        <h3 className="text-base font-bold text-slate-800">
          Reconstructed Temperature Profile
        </h3>
        <p className="text-xs text-slate-500 max-w-sm mt-1 mb-4 leading-relaxed">
          Select coordinates and an observation date, then click <strong>Reconstruct Temperature</strong> to run the OceanEmbed v3 model via live backend inference.
        </p>
        <div className="text-[11px] font-mono text-slate-500 border border-dashed border-slate-300 bg-slate-50 px-3 py-1.5 rounded">
          Default Coordinates: 15.00°N, 85.00°E (Bay of Bengal) • 2020-09-15
        </div>
      </div>
    );
  }

  const { depths_m, temperatures_c } = prediction;

  // Build depth points
  const points: DepthPoint[] = depths_m.map((depth, idx) => ({
    depth_m: depth,
    temperature_c: temperatures_c[idx],
    isThermocline: depth >= 75 && depth <= 150,
  }));

  // SVG Chart Dimensions
  const svgWidth = 560;
  const svgHeight = 440;
  const margin = { top: 30, right: 30, bottom: 40, left: 60 };
  const innerWidth = svgWidth - margin.left - margin.right;
  const innerHeight = svgHeight - margin.top - margin.bottom;

  // Non-linear depth mapping (expanded in upper 0-200m to clearly visualize thermocline structure)
  // Maps 0-1000m to 0-innerHeight pixels
  const depthToY = (depth: number) => {
    // Piecewise scale: 0-200m occupies 55% of height; 200-1000m occupies 45% of height
    if (depth <= 200) {
      return (depth / 200) * (innerHeight * 0.55);
    }
    return innerHeight * 0.55 + ((depth - 200) / 800) * (innerHeight * 0.45);
  };

  // Temperature to X coordinate
  const tempToX = (temp: number) => {
    const clampedT = Math.max(T_MIN, Math.min(T_MAX, temp));
    return ((clampedT - T_MIN) / (T_MAX - T_MIN)) * innerWidth;
  };

  // Generate SVG polyline path
  const pathD = points
    .map((pt, i) => {
      const x = tempToX(pt.temperature_c);
      const y = depthToY(pt.depth_m);
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    })
    .join(' ');

  // Thermocline band coordinates (75m to 150m)
  const thermoTopY = depthToY(75);
  const thermoBottomY = depthToY(150);
  const thermoHeight = thermoBottomY - thermoTopY;

  // Temperature ticks (4°C intervals: 6, 10, 14, 18, 22, 26, 30°C)
  const tempTicks = [6, 10, 14, 18, 22, 26, 30];

  // Depth ticks
  const depthTicks = [0, 50, 100, 150, 200, 300, 500, 700, 1000];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-200 mb-3 gap-2">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">
              Vertical Temperature Profile
            </h2>
            <span className="text-[11px] font-mono bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded border border-emerald-200">
              Real Inference
            </span>
          </div>
          <p className="text-xs text-slate-500">
            Depth increases downward (0m surface to 1000m deep ocean). Click any point to inspect.
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-xs text-slate-600">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-sky-700 inline-block" />
            <span className="font-mono text-[11px]">Predicted T (°C)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 bg-amber-100 border border-amber-300 rounded-xs inline-block" />
            <span className="font-mono text-[11px]">Thermocline (75–150m)</span>
          </div>
        </div>
      </div>

      {/* SVG Chart Container */}
      <div className="relative overflow-x-auto">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full max-w-full h-auto block select-none"
        >
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Thermocline Highlight Band (75m - 150m) */}
            <rect
              x="0"
              y={thermoTopY}
              width={innerWidth}
              height={thermoHeight}
              fill="#fef3c7"
              opacity="0.45"
            />
            <text
              x={innerWidth - 6}
              y={thermoTopY + 14}
              textAnchor="end"
              fill="#b45309"
              fontSize="10"
              fontFamily="monospace"
              fontWeight="600"
            >
              Thermocline Gradient Zone (75–150m)
            </text>

            {/* Vertical Gridlines (Temperature) */}
            {tempTicks.map((t) => {
              const x = tempToX(t);
              return (
                <g key={`grid-t-${t}`}>
                  <line
                    x1={x}
                    y1="0"
                    x2={x}
                    y2={innerHeight}
                    stroke="#e2e8f0"
                    strokeWidth="0.8"
                    strokeDasharray="3,3"
                  />
                  {/* Bottom X-axis label */}
                  <text
                    x={x}
                    y={innerHeight + 18}
                    textAnchor="middle"
                    fill="#64748b"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    {t}°
                  </text>
                </g>
              );
            })}

            {/* Horizontal Gridlines (Depth) */}
            {depthTicks.map((d) => {
              const y = depthToY(d);
              return (
                <g key={`grid-d-${d}`}>
                  <line
                    x1="0"
                    y1={y}
                    x2={innerWidth}
                    y2={y}
                    stroke="#e2e8f0"
                    strokeWidth="0.8"
                    strokeDasharray="3,3"
                  />
                  {/* Left Y-axis label */}
                  <text
                    x="-10"
                    y={y + 3}
                    textAnchor="end"
                    fill="#64748b"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    {d}m
                  </text>
                </g>
              );
            })}

            {/* Axis Labels */}
            <text
              x={innerWidth / 2}
              y={innerHeight + 34}
              textAnchor="middle"
              fill="#334155"
              fontSize="11"
              fontWeight="bold"
            >
              Temperature (°C)
            </text>
            <text
              x={-innerHeight / 2}
              y="-42"
              transform="rotate(-90)"
              textAnchor="middle"
              fill="#334155"
              fontSize="11"
              fontWeight="bold"
            >
              Depth (meters, non-linear)
            </text>

            {/* Outer Plot Box */}
            <rect
              x="0"
              y="0"
              width={innerWidth}
              height={innerHeight}
              fill="none"
              stroke="#cbd5e1"
              strokeWidth="1"
            />

            {/* Reconstructed Profile Curve */}
            <path
              d={pathD}
              fill="none"
              stroke="#0369a1"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Data Points on 15 Standard Depths */}
            {points.map((pt) => {
              const cx = tempToX(pt.temperature_c);
              const cy = depthToY(pt.depth_m);
              const isSelected = selectedDepth === pt.depth_m;
              const isHovered = hoveredPoint?.depth_m === pt.depth_m;

              return (
                <g
                  key={`pt-${pt.depth_m}`}
                  className="cursor-pointer"
                  onClick={() => onSelectDepth(pt.depth_m)}
                  onMouseEnter={() => setHoveredPoint(pt)}
                  onMouseLeave={() => setHoveredPoint(null)}
                >
                  {/* Outer ring for selected or hovered depth */}
                  {(isSelected || isHovered) && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r="9"
                      fill="none"
                      stroke={isSelected ? '#0284c7' : '#94a3b8'}
                      strokeWidth="1.5"
                      strokeDasharray="2,2"
                    />
                  )}
                  {/* Data dot */}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isSelected ? '5.5' : '4'}
                    fill={pt.isThermocline ? '#d97706' : '#0369a1'}
                    stroke="#ffffff"
                    strokeWidth="1.5"
                  />
                </g>
              );
            })}

            {/* Active Hover Tooltip Overlay */}
            {hoveredPoint && (
              <g
                transform={`translate(${tempToX(hoveredPoint.temperature_c)}, ${depthToY(hoveredPoint.depth_m)})`}
                className="pointer-events-none"
              >
                <rect
                  x="12"
                  y="-22"
                  width="110"
                  height="34"
                  rx="4"
                  fill="#0f172a"
                  opacity="0.92"
                />
                <text x="18" y="-9" fill="#f8fafc" fontSize="10" fontWeight="bold">
                  Depth: {hoveredPoint.depth_m} m
                </text>
                <text x="18" y="5" fill="#38bdf8" fontSize="10" fontFamily="monospace">
                  Temp: {hoveredPoint.temperature_c.toFixed(2)} °C
                </text>
              </g>
            )}
          </g>
        </svg>
      </div>

      {/* Quick Footnote on Depth Profile Physics */}
      <div className="mt-2 pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
        <span>
          Surface: <strong className="text-slate-700">{temperatures_c[0].toFixed(2)}°C</strong> | 100m: <strong className="text-slate-700">{temperatures_c[7].toFixed(2)}°C</strong> | 1000m: <strong className="text-slate-700">{temperatures_c[14].toFixed(2)}°C</strong>
        </span>
        <span className="font-mono text-slate-400">
          Inference: {prediction.inference_ms} ms
        </span>
      </div>
    </div>
  );
};
