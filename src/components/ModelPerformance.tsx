import React from 'react';
import { Award, CheckCircle2, TrendingDown, Clock, ShieldCheck, Database } from 'lucide-react';

export const ModelPerformance: React.FC = () => {
  // Audited depth-wise ARGO RMSE table (September 2020 contemporaneous match)
  const depthBreakdown = [
    { depth: '5m', cnn: 0.58, v3: 0.53, delta: '-0.05°C', thermo: false },
    { depth: '10m', cnn: 0.76, v3: 0.60, delta: '-0.16°C', thermo: false },
    { depth: '20m', cnn: 0.87, v3: 0.66, delta: '-0.21°C', thermo: false },
    { depth: '30m', cnn: 0.96, v3: 0.72, delta: '-0.24°C', thermo: false },
    { depth: '50m', cnn: 1.04, v3: 0.87, delta: '-0.17°C', thermo: false },
    { depth: '75m', cnn: 1.22, v3: 1.04, delta: '-0.18°C', thermo: true },
    { depth: '100m', cnn: 1.69, v3: 1.53, delta: '-0.16°C', thermo: true },
    { depth: '125m', cnn: 1.20, v3: 1.06, delta: '-0.14°C', thermo: true },
    { depth: '150m', cnn: 1.03, v3: 0.86, delta: '-0.16°C', thermo: true },
    { depth: '200m', cnn: 1.11, v3: 0.91, delta: '-0.20°C', thermo: false },
    { depth: '300m', cnn: 0.61, v3: 0.36, delta: '-0.26°C', thermo: false },
    { depth: '500m', cnn: 0.45, v3: 0.30, delta: '-0.15°C', thermo: false },
    { depth: '700m', cnn: 0.38, v3: 0.26, delta: '-0.11°C', thermo: false },
    { depth: '1000m', cnn: 0.40, v3: 0.26, delta: '-0.15°C', thermo: false },
  ];

  return (
    <section className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 mb-5 gap-2">
        <div>
          <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Award className="w-5 h-5 text-sky-700" />
            Validation Results
            <span className="text-[10px] font-mono font-semibold bg-slate-100 text-slate-600 border border-slate-200 px-2 py-0.5 rounded normal-case tracking-normal">
              Previously evaluated benchmarks
            </span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Two distinct benchmarks: the <strong>GLORYS held-out reference</strong> (supervised
            training/reference dataset) and the <strong>ARGO observational benchmark</strong>{' '}
            (in-situ floats). The figures on this card are historical, verified experiment results
            — they are not recomputed per click. Live, request-time ARGO values are shown in the{' '}
            <a href="#argo-validation" className="text-sky-700 underline underline-offset-2">
              ARGO Observational Validation
            </a>{' '}
            section above.
          </p>
        </div>
      </div>

      {/* Top 4 Performance Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5 mb-6">
        {/* Metric 1: GLORYS Reanalysis Reference */}
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
          <div className="text-[11px] font-mono text-slate-500 uppercase tracking-wider mb-1">
            GLORYS Held-Out Benchmark
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-extrabold font-mono text-slate-900">0.8601°C</span>
            <span className="text-xs font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
              -17.4% error
            </span>
          </div>
          <div className="mt-2 text-[11px] text-slate-500 flex items-center justify-between border-t border-slate-200 pt-1.5 font-mono">
            <span>Simple CNN: 1.0418°C</span>
            <span className="text-slate-400">Historical • Sep 2020 test</span>
          </div>
        </div>

        {/* Metric 2: ARGO Observational Check */}
        <div className="p-4 bg-amber-50/50 border border-amber-200 rounded-lg">
          <div className="text-[11px] font-mono text-amber-800 uppercase tracking-wider mb-1">
            ARGO Observational Benchmark
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-extrabold font-mono text-sky-900">0.7973°C</span>
            <span className="text-xs font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
              -16.3% error
            </span>
          </div>
          <div className="mt-2 text-[11px] text-slate-500 flex items-center justify-between border-t border-amber-200 pt-1.5 font-mono">
            <span>Simple CNN: 0.9526°C</span>
            <span className="text-slate-400">Historical • Sep 2020</span>
          </div>
        </div>

        {/* Metric 3: Target Resolution & Strata */}
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
          <div className="text-[11px] font-mono text-slate-500 uppercase tracking-wider mb-1">
            Target Resolution & Depths
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-extrabold font-mono text-slate-900">15 Depths</span>
            <span className="text-xs font-semibold text-slate-600 bg-slate-200/60 px-1.5 py-0.5 rounded">
              0–1000 m
            </span>
          </div>
          <div className="mt-2 text-[11px] text-slate-500 flex items-center justify-between border-t border-slate-200 pt-1.5 font-mono">
            <span>0.25° × 0.25° Grid</span>
            <span className="text-slate-400">101 × 241</span>
          </div>
        </div>

        {/* Metric 4: Measured API Latency Benchmark */}
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
          <div className="text-[11px] font-mono text-slate-500 uppercase tracking-wider mb-1">
            Backend Benchmark (CPU)
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-extrabold font-mono text-slate-900">~35.9 ms</span>
            <span className="text-xs font-semibold text-slate-600 bg-slate-200/60 px-1.5 py-0.5 rounded">
              Mean API Latency
            </span>
          </div>
          <div className="mt-2 text-[11px] text-slate-500 flex items-center justify-between border-t border-slate-200 pt-1.5 font-mono">
            <span>Median: 34.9 ms</span>
            <span className="text-slate-400">Neural: 31.9 ms</span>
          </div>
        </div>
      </div>

      {/* Observational Check Notice */}
      <div className="bg-sky-50/60 border border-sky-200 rounded-lg p-3 text-xs text-sky-900 mb-6 flex items-start gap-2.5">
        <ShieldCheck className="w-4 h-4 text-sky-700 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold block">Observational Verification Protocol (historical benchmark):</span>
          <span>
            Independent observational check using <strong>497 matched profile-depth observations from 36 ARGO profiles across 28 unique WMO floats in September 2020</strong>.
            These are static results from a previously evaluated, verified offline experiment — the
            live per-request equivalent is served by <code className="font-mono">GET /argo/summary</code> in the ARGO section above.
          </span>
        </div>
      </div>

      {/* Depth-Wise Comparison Table (Highlighting Thermocline) */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
            ARGO Depth-wise RMSE — Previously evaluated benchmark
          </h3>
          <div className="flex items-center gap-2 text-[11px] text-slate-500">
            <span className="w-2.5 h-2.5 rounded-xs bg-amber-100 border border-amber-300 inline-block" />
            <span>Thermocline-Region ARGO RMSE</span>
          </div>
        </div>

        <div className="overflow-x-auto border border-slate-200 rounded-lg">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 border-b border-slate-200 text-[11px] font-mono text-slate-600 uppercase">
              <tr>
                <th className="py-2 px-3">Depth</th>
                <th className="py-2 px-3">Simple CNN Baseline</th>
                <th className="py-2 px-3">OceanEmbed v3</th>
                <th className="py-2 px-3">Error Reduction</th>
                <th className="py-2 px-3">Relative Gain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {depthBreakdown.map((row) => (
                <tr
                  key={row.depth}
                  className={row.thermo ? 'bg-amber-50/40 hover:bg-amber-50/70' : 'hover:bg-slate-50'}
                >
                  <td className="py-1.5 px-3 font-semibold text-slate-800 flex items-center gap-1.5">
                    {row.depth}
                    {row.thermo && (
                      <span className="text-[9px] font-sans bg-amber-200/80 text-amber-900 px-1 py-0.2 rounded">
                        thermocline
                      </span>
                    )}
                  </td>
                  <td className="py-1.5 px-3 text-slate-600">{row.cnn.toFixed(2)} °C</td>
                  <td className="py-1.5 px-3 font-bold text-sky-800">{row.v3.toFixed(2)} °C</td>
                  <td className="py-1.5 px-3 font-semibold text-emerald-700">{row.delta}</td>
                  <td className="py-1.5 px-3 text-emerald-700">
                    {(((row.cnn - row.v3) / row.cnn) * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-[10px] text-slate-500 mt-1.5 font-mono">
          * Historical, previously evaluated benchmark: OceanEmbed v3 achieved lower ARGO RMSE than the Simple CNN at all 14 evaluated depth levels (0m had no ARGO observations). Live depth-wise values are returned by <code>GET /argo/summary</code> in the ARGO section.
        </p>
      </div>
    </section>
  );
};
