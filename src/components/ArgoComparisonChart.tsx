import React, { useState } from 'react';
import { LineChart, GitCompareArrows, Info } from 'lucide-react';
import { ArgoProfileComparison, DepthComparison } from '../types';

interface ArgoComparisonChartProps {
  comparison: ArgoProfileComparison | null;
}

// Thermocline region highlighted in the comparison chart (75–200 m)
const THERMOCLINE_TOP_M = 75;
const THERMOCLINE_BOTTOM_M = 200;

const OBSERVED_COLOR = '#d97706'; // ARGO observed — amber
const PREDICTED_COLOR = '#0369a1'; // OceanEmbed predicted — sky blue

/**
 * Scientific core of ARGO Validation Mode: the authentic observed ARGO
 * temperature profile against the OceanEmbed v3 reconstruction produced by
 * the backend for the same location and date. Only valid matched depths are
 * plotted — missing observations are never imputed in the browser.
 */
export const ArgoComparisonChart: React.FC<ArgoComparisonChartProps> = ({ comparison }) => {
  const [hoveredRow, setHoveredRow] = useState<DepthComparison | null>(null);

  if (!comparison) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs flex flex-col items-center justify-center min-h-[320px] text-center">
        <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 mb-3">
          <LineChart className="w-6 h-6" />
        </div>
        <h3 className="text-base font-bold text-slate-800">Observed vs Predicted Profile</h3>
        <p className="text-xs text-slate-500 max-w-sm mt-1 leading-relaxed">
          No ARGO profile selected. Choose a float on the map or from the profile list to run a
          live backend comparison.
        </p>
      </div>
    );
  }

  const { depth_comparison: rows, argo_profile: profile } = comparison;

  const observedRows = rows.filter((r) => r.observed && r.observed_c !== null);
  const predictedRows = rows.filter((r) => r.predicted_c !== null);

  // Dynamic temperature domain derived ONLY from the backend-returned values
  const temps = [
    ...observedRows.map((r) => r.observed_c as number),
    ...predictedRows.map((r) => r.predicted_c as number),
  ];

  // Defensive guard: never fabricate an axis when the backend returned no values
  if (temps.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs text-center min-h-[240px] flex flex-col items-center justify-center">
        <h3 className="text-sm font-bold text-slate-800">Observed vs Predicted Profile</h3>
        <p className="text-xs text-slate-500 mt-1">
          The backend returned no plottable values for this profile.
        </p>
      </div>
    );
  }

  const tMin = Math.floor(Math.min(...temps) - 1);
  const tMax = Math.ceil(Math.max(...temps) + 1);

  const svgWidth = 560;
  const svgHeight = 430;
  const margin = { top: 26, right: 24, bottom: 42, left: 62 };
  const innerWidth = svgWidth - margin.left - margin.right;
  const innerHeight = svgHeight - margin.top - margin.bottom;

  // Same piecewise depth mapping as Reconstruction Mode (0–200 m expanded)
  const depthToY = (depth: number) => {
    if (depth <= 200) {
      return (depth / 200) * (innerHeight * 0.55);
    }
    return innerHeight * 0.55 + ((depth - 200) / 800) * (innerHeight * 0.45);
  };

  const tempToX = (temp: number) => {
    const span = Math.max(tMax - tMin, 1);
    return ((temp - tMin) / span) * innerWidth;
  };

  const buildPath = (data: DepthComparison[], valueKey: 'observed_c' | 'predicted_c') =>
    data
      .map((row, i) => {
        const v = row[valueKey];
        if (v === null) return '';
        return `${i === 0 ? 'M' : 'L'} ${tempToX(v)} ${depthToY(row.depth_m)}`;
      })
      .join(' ');

  // Temperature ticks (adaptive step, whole-degree labels)
  const rawStep = (tMax - tMin) / 6;
  const step = rawStep <= 2 ? 2 : rawStep <= 5 ? 5 : 10;
  const tempTicks: number[] = [];
  for (let t = Math.ceil(tMin / step) * step; t <= tMax; t += step) tempTicks.push(t);

  const depthTicks = [0, 50, 100, 150, 200, 300, 500, 700, 1000];

  const thermoTopY = depthToY(THERMOCLINE_TOP_M);
  const thermoBottomY = depthToY(THERMOCLINE_BOTTOM_M);

  // ---- Depth-wise error chart data -------------------------------------
  const errorRows = rows.filter(
    (r): r is DepthComparison & { error_c: number } => r.error_c !== null
  );
  const maxAbsError = Math.max(0.25, ...errorRows.map((r) => Math.abs(r.error_c)));
  const errSvgWidth = 560;
  const errRowHeight = 17;
  const errMargin = { top: 26, right: 24, bottom: 26, left: 62 };
  const errInnerWidth = errSvgWidth - errMargin.left - errMargin.right;
  const errSvgHeight = errMargin.top + errMargin.bottom + errorRows.length * errRowHeight;
  const errZeroX = errMargin.left + errInnerWidth / 2;
  const errScale = (v: number) => (v / maxAbsError) * (errInnerWidth / 2);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-xs">
      {/* Header + Legend */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 pb-3 border-b border-slate-100 mb-3">
        <div>
          <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <GitCompareArrows className="w-4 h-4 text-sky-700" />
            Observed vs Predicted Profile
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            WMO {profile.wmo} • {profile.date} • {profile.latitude.toFixed(2)}°N,{' '}
            {profile.longitude.toFixed(2)}°E • matched date {comparison.matched_model_date}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-[11px] font-semibold shrink-0">
          <span className="flex items-center gap-1.5 text-amber-800">
            <span className="w-4 h-0.5 rounded" style={{ background: OBSERVED_COLOR }} />
            ARGO observed
          </span>
          <span className="flex items-center gap-1.5 text-sky-900">
            <span className="w-4 h-0.5 rounded" style={{ background: PREDICTED_COLOR }} />
            OceanEmbed v3
          </span>
          <span className="flex items-center gap-1.5 text-amber-700">
            <span className="w-4 h-3 rounded-xs bg-amber-100 border border-amber-300 inline-block" />
            Thermocline {THERMOCLINE_TOP_M}–{THERMOCLINE_BOTTOM_M} m
          </span>
        </div>
      </div>

      {/* Main depth–temperature chart (depth increases downward) */}
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full h-auto max-w-[640px] mx-auto block"
        >
          <rect
            x={margin.left}
            y={margin.top}
            width={innerWidth}
            height={innerHeight}
            fill="#f8fafc"
            stroke="#e2e8f0"
          />

          {/* Thermocline band */}
          <rect
            x={margin.left}
            y={margin.top + thermoTopY}
            width={innerWidth}
            height={thermoBottomY - thermoTopY}
            fill="#fef3c7"
            opacity="0.55"
          />
          <text
            x={margin.left + innerWidth - 6}
            y={margin.top + thermoTopY + 13}
            textAnchor="end"
            fill="#b45309"
            fontSize="9.5"
            fontWeight="600"
          >
            thermocline region
          </text>

          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Grid lines + depth axis */}
            {depthTicks.map((d) => (
              <g key={`dt-${d}`}>
                <line
                  x1={0}
                  y1={depthToY(d)}
                  x2={innerWidth}
                  y2={depthToY(d)}
                  stroke="#e2e8f0"
                  strokeWidth="1"
                />
                <text
                  x={-8}
                  y={depthToY(d) + 3.5}
                  textAnchor="end"
                  fill="#64748b"
                  fontSize="10"
                  fontFamily="monospace"
                >
                  {d}
                </text>
              </g>
            ))}

            {/* Temperature axis ticks */}
            {tempTicks.map((t) => (
              <g key={`tt-${t}`}>
                <line
                  x1={tempToX(t)}
                  y1={0}
                  x2={tempToX(t)}
                  y2={innerHeight}
                  stroke="#e2e8f0"
                  strokeWidth="1"
                />
                <text
                  x={tempToX(t)}
                  y={innerHeight + 16}
                  textAnchor="middle"
                  fill="#64748b"
                  fontSize="10"
                  fontFamily="monospace"
                >
                  {t}
                </text>
              </g>
            ))}

            <text
              x={innerWidth / 2}
              y={innerHeight + 34}
              textAnchor="middle"
              fill="#475569"
              fontSize="11"
              fontWeight="600"
            >
              Temperature (°C)
            </text>
            <text
              transform={`translate(-46, ${innerHeight / 2}) rotate(-90)`}
              textAnchor="middle"
              fill="#475569"
              fontSize="11"
              fontWeight="600"
            >
              Depth (m)
            </text>

            {/* OceanEmbed predicted curve (all 15 target depths) */}
            <path
              d={buildPath(predictedRows, 'predicted_c')}
              fill="none"
              stroke={PREDICTED_COLOR}
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* ARGO observed curve (genuinely observed matched depths only) */}
            <path
              d={buildPath(observedRows, 'observed_c')}
              fill="none"
              stroke={OBSERVED_COLOR}
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Predicted points */}
            {predictedRows.map((row) => (
              <circle
                key={`p-${row.depth_m}`}
                cx={tempToX(row.predicted_c as number)}
                cy={depthToY(row.depth_m)}
                r="3.4"
                fill={PREDICTED_COLOR}
                stroke="#ffffff"
                strokeWidth="1.2"
              />
            ))}

            {/* Observed points (interactive) */}
            {observedRows.map((row) => (
              <g
                key={`o-${row.depth_m}`}
                className="cursor-pointer"
                onMouseEnter={() => setHoveredRow(row)}
                onMouseLeave={() => setHoveredRow(null)}
              >
                <circle
                  cx={tempToX(row.observed_c as number)}
                  cy={depthToY(row.depth_m)}
                  r="7"
                  fill="transparent"
                />
                <circle
                  cx={tempToX(row.observed_c as number)}
                  cy={depthToY(row.depth_m)}
                  r={hoveredRow?.depth_m === row.depth_m ? 5 : 3.8}
                  fill={OBSERVED_COLOR}
                  stroke="#ffffff"
                  strokeWidth="1.3"
                />
              </g>
            ))}

            {/* Hover tooltip */}
            {hoveredRow && hoveredRow.observed_c !== null && hoveredRow.predicted_c !== null && (
              <g
                transform={`translate(${tempToX(hoveredRow.observed_c)}, ${depthToY(hoveredRow.depth_m)})`}
                className="pointer-events-none"
              >
                <rect x="12" y="-44" width="176" height="58" rx="4" fill="#0f172a" opacity="0.94" />
                <text x="20" y="-30" fill="#f8fafc" fontSize="10" fontWeight="bold">
                  Depth {hoveredRow.depth_m} m
                </text>
                <text x="20" y="-16" fill="#fdba74" fontSize="10" fontFamily="monospace">
                  ARGO: {hoveredRow.observed_c.toFixed(2)} °C
                </text>
                <text x="20" y="-2" fill="#7dd3fc" fontSize="10" fontFamily="monospace">
                  Model: {hoveredRow.predicted_c.toFixed(2)} °C | Δ{' '}
                  {hoveredRow.error_c !== null ? hoveredRow.error_c.toFixed(2) : '—'}
                </text>
              </g>
            )}
          </g>
        </svg>
      </div>

      {/* Depth-wise error visualization */}
      <div className="mt-3 pt-3 border-t border-slate-100">
        <div className="flex items-center justify-between mb-1.5">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800">
            Depth-wise Error (Model − ARGO)
          </h4>
          <span className="text-[10px] text-slate-500 font-mono">
            {errorRows.length} matched depths
          </span>
        </div>

        {errorRows.length === 0 ? (
          <p className="text-xs text-slate-500 py-3 text-center">
            No matched depth observations were returned for this profile.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <svg
              viewBox={`0 0 ${errSvgWidth} ${errSvgHeight}`}
              className="w-full h-auto max-w-[640px] mx-auto block"
            >
              {/* Zero line */}
              <line
                x1={errZeroX}
                y1={errMargin.top - 6}
                x2={errZeroX}
                y2={errSvgHeight - errMargin.bottom + 6}
                stroke="#94a3b8"
                strokeWidth="1"
              />
              <text
                x={errZeroX}
                y={errMargin.top - 11}
                textAnchor="middle"
                fill="#64748b"
                fontSize="9"
                fontFamily="monospace"
              >
                0 °C
              </text>
              <text x={errMargin.left} y={errSvgHeight - 8} fill="#64748b" fontSize="9" fontFamily="monospace">
                −{maxAbsError.toFixed(2)}
              </text>
              <text
                x={errMargin.left + errInnerWidth}
                y={errSvgHeight - 8}
                textAnchor="end"
                fill="#64748b"
                fontSize="9"
                fontFamily="monospace"
              >
                +{maxAbsError.toFixed(2)}
              </text>

              {errorRows.map((row, idx) => {
                const y = errMargin.top + idx * errRowHeight;
                const w = errScale(row.error_c);
                const isThermo =
                  row.depth_m >= THERMOCLINE_TOP_M && row.depth_m <= THERMOCLINE_BOTTOM_M;
                const barEnd = w >= 0 ? errZeroX + w : errZeroX + w;
                return (
                  <g key={`err-${row.depth_m}`}>
                    <text
                      x={errMargin.left - 8}
                      y={y + errRowHeight * 0.72}
                      textAnchor="end"
                      fill={isThermo ? '#b45309' : '#475569'}
                      fontSize="9.5"
                      fontFamily="monospace"
                    >
                      {row.depth_m}m
                    </text>
                    <rect
                      x={w >= 0 ? errZeroX : barEnd}
                      y={y + 2}
                      width={Math.max(Math.abs(w), 1)}
                      height={errRowHeight - 6}
                      fill={row.error_c >= 0 ? '#0284c7' : '#f43f5e'}
                      opacity={isThermo ? 1 : 0.75}
                      rx="1.5"
                    />
                    <text
                      x={w >= 0 ? barEnd + 4 : barEnd - 4}
                      y={y + errRowHeight * 0.72}
                      textAnchor={w >= 0 ? 'start' : 'end'}
                      fill="#475569"
                      fontSize="9"
                      fontFamily="monospace"
                    >
                      {row.error_c >= 0 ? '+' : ''}
                      {row.error_c.toFixed(2)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        )}

        <p className="text-[10px] text-slate-500 mt-1.5 flex items-start gap-1">
          <Info className="w-3 h-3 shrink-0 mt-0.5" />
          <span>
            Only depths with a genuine ARGO observation are plotted. Depth {THERMOCLINE_TOP_M}–
            {THERMOCLINE_BOTTOM_M} m is the thermocline region.
          </span>
        </p>
      </div>
    </div>
  );
};


