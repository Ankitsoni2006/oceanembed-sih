import React from 'react';
import {
  ArrowRight,
  Database,
  Droplets,
  Gauge,
  Layers,
  Lightbulb,
  Rocket,
  Target,
  Waves,
} from 'lucide-react';

/**
 * Concise scientific pipeline + uniqueness story.
 * Framing rules honoured here:
 *  - GLORYS = supervised training/reference dataset (NOT "ground truth")
 *  - ARGO   = observational evaluation dataset (NOT a model input)
 *  - No claims of SOTA, operational forecasting, real-time global deployment,
 *    or cyclone prediction.
 */
const PIPELINE_STAGES = [
  {
    icon: Droplets,
    title: 'Surface observations',
    detail: '7 daily satellite surface fields + 7 quality masks on a 0.25° grid',
    tone: 'sky',
  },
  {
    icon: Layers,
    title: 'OceanEmbed v3',
    detail: 'U-Net encoder + parallel multi-depth decoder, 1.28M parameters',
    tone: 'sky',
  },
  {
    icon: Waves,
    title: '15-depth subsurface temperature',
    detail: '0–1000 m reconstructed column at standard levels',
    tone: 'sky',
  },
  {
    icon: Target,
    title: 'ARGO observational comparison',
    detail: 'Independent in-situ floats, evaluation only — never an input',
    tone: 'amber',
  },
  {
    icon: Gauge,
    title: 'RMSE / MAE / Bias',
    detail: 'Per-profile and aggregate error, computed at request time',
    tone: 'emerald',
  },
];

const UNIQUE_POINTS = [
  {
    icon: Layers,
    title: 'Surface-to-subsurface at 15 depths',
    body: 'One forward pass reconstructs the full 0–1000 m temperature column at 15 standard levels.',
  },
  {
    icon: Database,
    title: 'Broad surface coverage vs sparse subsurface truth',
    body: 'Satellite surface fields cover the basin daily, while direct subsurface measurements remain sparse in space and time.',
  },
  {
    icon: Target,
    title: 'Explicit ARGO observational evaluation',
    body: 'Every claim is checked against authentic profiling-float observations, not only against the training reference.',
  },
  {
    icon: Gauge,
    title: 'Quantified uncertainty per profile',
    body: 'RMSE, MAE, Bias and Pearson r are reported for each float and depth, so error is visible rather than implied.',
  },
  {
    icon: Droplets,
    title: 'Thermocline-aware analysis',
    body: 'The 75–200 m thermocline region is tracked explicitly — the hardest and most dynamic part of the column.',
  },
  {
    icon: Rocket,
    title: 'Fast single-point inference',
    body: 'The deployed v3 checkpoint answers a location/date query through the API in milliseconds on CPU.',
  },
];

export const WhyOceanEmbed: React.FC = () => {
  return (
    <section className="space-y-6">
      {/* ---------- Scientific pipeline ---------- */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 shadow-xs">
        <div className="pb-4 border-b border-slate-200 mb-5">
          <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-sky-700" />
            Reconstruction &amp; Evaluation Pipeline
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            What the model consumes, what it produces, and how it is independently evaluated.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 mb-5">
          {PIPELINE_STAGES.map((stage, idx) => {
            const Icon = stage.icon;
            const border =
              stage.tone === 'amber'
                ? 'border-amber-300 bg-amber-50/60'
                : stage.tone === 'emerald'
                  ? 'border-emerald-300 bg-emerald-50/60'
                  : 'border-slate-200 bg-slate-50';
            const iconColor =
              stage.tone === 'amber'
                ? 'text-amber-700'
                : stage.tone === 'emerald'
                  ? 'text-emerald-700'
                  : 'text-sky-700';
            return (
              <React.Fragment key={stage.title}>
                <div className={`p-3.5 border rounded-lg flex flex-col ${border}`}>
                  <div className="flex items-center justify-between mb-2">
                    <Icon className={`w-4 h-4 ${iconColor}`} />
                    <span className="text-[10px] font-mono text-slate-500">
                      {String(idx + 1).padStart(2, '0')}
                    </span>
                  </div>
                  <h3 className="text-xs font-bold text-slate-800 leading-snug">{stage.title}</h3>
                  <p className="text-[11px] text-slate-600 leading-relaxed mt-1">{stage.detail}</p>
                </div>
                {idx < PIPELINE_STAGES.length - 1 && (
                  <div className="hidden md:flex items-center justify-center text-slate-400">
                    <ArrowRight className="w-4 h-4 md:rotate-90 lg:rotate-0" />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="font-bold text-slate-800 mb-1.5 flex items-center gap-1.5">
              <Database className="w-4 h-4 text-sky-700" />
              GLORYS — training / reference dataset
            </div>
            <p className="text-slate-600 leading-relaxed">
              GLORYS reanalysis is the <strong>supervised training and reference target</strong>.
              It is a reanalysis product, <strong>not observational ground truth</strong>.
            </p>
          </div>
          <div className="p-3.5 bg-amber-50/60 border border-amber-200 rounded-lg">
            <div className="font-bold text-amber-950 mb-1.5 flex items-center gap-1.5">
              <Waves className="w-4 h-4 text-amber-700" />
              ARGO — observational evaluation dataset
            </div>
            <p className="text-amber-900 leading-relaxed">
              ARGO floats are <strong>only used to evaluate</strong> the reconstruction against
              direct in-situ measurements. ARGO temperature is <strong>never a model input</strong>,
              and this prototype does not provide a live or global ARGO feed.
            </p>
          </div>
        </div>
      </div>

      {/* ---------- Uniqueness / value proposition ---------- */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 shadow-xs">
        <div className="pb-4 border-b border-slate-200 mb-5">
          <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-amber-600" />
            Why OceanEmbed? — What is unique
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Defensible differentiators of this research prototype.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
          {UNIQUE_POINTS.map((point) => {
            const Icon = point.icon;
            return (
              <div
                key={point.title}
                className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg"
              >
                <div className="font-bold text-slate-800 mb-1.5 flex items-center gap-1.5 text-xs">
                  <Icon className="w-4 h-4 text-sky-700 shrink-0" />
                  {point.title}
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">{point.body}</p>
              </div>
            );
          })}
        </div>

        <p className="text-[10px] text-slate-500 mt-4 font-mono leading-relaxed">
          Scope: research prototype for the North Indian Ocean, Jan–Sep 2020 processed dataset. Not
          an operational forecasting system, not a real-time global monitoring service.
        </p>
      </div>
    </section>
  );
};
