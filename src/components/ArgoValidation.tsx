import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Database,
  RefreshCw,
  Ruler,
  Search,
  SearchX,
  Waves,
} from 'lucide-react';
import { api } from '../services/api';
import {
  ArgoProfile,
  ArgoProfileComparison,
  ArgoProfileListResponse,
  ArgoSummary,
} from '../types';
import { OceanMap } from './OceanMap';
import { ArgoComparisonChart } from './ArgoComparisonChart';
import { THERMOCLINE_TOP_M, THERMOCLINE_BOTTOM_M, isInThermoclineBand } from '../lib/ocean';

/**
 * The ARGO catalog is static metadata (~profiles), so it is cached once per
 * browser session to avoid refetching when the judge toggles modes. No NetCDF
 * data and no model weights ever reach the browser — JSON from FastAPI only.
 */
let profilesCache: ArgoProfileListResponse | null = null;

const EMPTY_SELECTION_PLACEHOLDER = (
  <div className="bg-white border border-dashed border-slate-300 rounded-xl p-8 shadow-xs flex flex-col items-center justify-center min-h-[300px] text-center">
    <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 mb-3">
      <SearchX className="w-6 h-6" />
    </div>
    <h3 className="text-base font-bold text-slate-800">No ARGO profile selected</h3>
    <p className="text-xs text-slate-500 max-w-sm mt-1 leading-relaxed">
      Click an ARGO marker on the map or pick a float from the profile list to run an
      on-demand OceanEmbed v3 comparison against the observed profile.
    </p>
  </div>
);

/**
 * ARGO Observational Validation Mode.
 *
 * Every displayed ARGO number is returned by the FastAPI backend at request
 * time (GET /argo/profiles, GET /argo/summary, GET /argo/compare/{id}).
 * ARGO observations are NEVER passed into OceanEmbed — the model consumes
 * surface observations only, and the backend evaluates the reconstruction
 * against the float afterwards.
 */
export const ArgoValidation: React.FC = () => {
  // --- Catalog state ---
  const [profilesData, setProfilesData] = useState<ArgoProfileListResponse | null>(profilesCache);
  const [profilesLoading, setProfilesLoading] = useState<boolean>(profilesCache === null);
  const [profilesError, setProfilesError] = useState<string | null>(null);

  // --- Aggregate summary state ---
  const [summary, setSummary] = useState<ArgoSummary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState<boolean>(true);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  // --- Selection & per-profile comparison state ---
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ArgoProfileComparison | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState<boolean>(false);
  const [comparisonError, setComparisonError] = useState<string | null>(null);

  // --- Profile list search ---
  const [search, setSearch] = useState<string>('');
  const comparisonSeq = useRef<number>(0);

  const loadProfiles = useCallback(async (force: boolean = false) => {
    if (profilesCache && !force) {
      setProfilesData(profilesCache);
      setProfilesLoading(false);
      setProfilesError(null);
      return;
    }
    setProfilesLoading(true);
    setProfilesError(null);
    try {
      const data = await api.getArgoProfiles();
      profilesCache = data;
      setProfilesData(data);
    } catch (err: any) {
      setProfilesData(null);
      setProfilesError(err.message || 'Backend unavailable — ARGO profiles could not be loaded.');
    } finally {
      setProfilesLoading(false);
    }
  }, []);

  const loadSummary = useCallback(async () => {
    setSummaryLoading(true);
    setSummaryError(null);
    try {
      const data = await api.getArgoSummary();
      setSummary(data);
    } catch (err: any) {
      setSummary(null);
      setSummaryError(err.message || 'Backend unavailable — aggregate ARGO evaluation unavailable.');
    } finally {
      setSummaryLoading(false);
    }
  }, []);

  // Initial load of catalog + aggregate evaluation
  useEffect(() => {
    loadProfiles();
    loadSummary();
  }, [loadProfiles, loadSummary]);

  // Load (or clear) the per-profile comparison whenever the selection changes.
  // Stale responses are discarded so a previous profile's data is never shown
  // under a newly selected profile.
  useEffect(() => {
    const seq = ++comparisonSeq.current;
    setComparison(null);
    setComparisonError(null);

    if (!selectedId) {
      setComparisonLoading(false);
      return;
    }

    setComparisonLoading(true);
    api
      .getArgoComparison(selectedId)
      .then((data) => {
        if (seq !== comparisonSeq.current) return;
        setComparison(data);
      })
      .catch((err: any) => {
        if (seq !== comparisonSeq.current) return;
        setComparison(null);
        setComparisonError(err.message || 'ARGO comparison unavailable for this profile.');
      })
      .finally(() => {
        if (seq !== comparisonSeq.current) return;
        setComparisonLoading(false);
      });
  }, [selectedId]);

  const handleSelectProfile = useCallback((profileId: string) => {
    setSelectedId((current) => (current === profileId ? current : profileId));
  }, []);

  // Searchable profile list (WMO, date, coordinates, profile id)
  const filteredProfiles = useMemo<ArgoProfile[]>(() => {
    const all = profilesData?.profiles ?? [];
    const q = search.trim().toLowerCase();
    if (!q) return all;
    return all.filter(
      (p) =>
        p.wmo.toLowerCase().includes(q) ||
        p.profile_id.toLowerCase().includes(q) ||
        p.date.includes(q) ||
        p.latitude.toFixed(2).includes(q) ||
        p.longitude.toFixed(2).includes(q)
    );
  }, [profilesData, search]);

  const selectedProfile = useMemo(
    () => profilesData?.profiles.find((p) => p.profile_id === selectedId) ?? null,
    [profilesData, selectedId]
  );

  // Live formatting helpers — every value originates from a backend response
  const fmt = (v: number | null | undefined, digits = 2, suffix = '') =>
    v === null || v === undefined || Number.isNaN(v) ? '—' : `${v.toFixed(digits)}${suffix}`;
  const signed = (v: number | null | undefined, digits = 2, suffix = '') =>
    v === null || v === undefined || Number.isNaN(v)
      ? '—'
      : `${v > 0 ? '+' : ''}${v.toFixed(digits)}${suffix}`;
  const completenessPct = comparison
    ? Math.round(
        (comparison.surface_input_availability.available_count /
          comparison.surface_input_availability.total_count) *
          100
      )
    : null;

  return (
    <section id="argo-validation" className="space-y-6">
      {/* ================= Section header ================= */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
          <div>
            <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <Waves className="w-5 h-5 text-amber-600" />
              ARGO Observational Validation
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Compare OceanEmbed reconstruction against real in-situ profiling-float observations.
            </p>
          </div>
          <span className="text-[11px] font-mono bg-amber-50 text-amber-900 border border-amber-200 px-2.5 py-1 rounded font-medium shrink-0">
            September 2020 Offline Observational Evaluation — Coriolis / INCOIS GDAC
          </span>
        </div>

        <div className="mt-3 bg-sky-50/60 border border-sky-200 rounded-lg p-3 text-xs text-sky-900 flex items-start gap-2.5">
          <Database className="w-4 h-4 text-sky-700 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold block">Evaluation-only data path:</span>
            <span>
              ARGO temperature is <strong>never a model input</strong>. The backend reconstructs the
              subsurface profile from surface observations alone, then evaluates it against the
              float. Statistics below are computed at request time — no value is hardcoded.
            </span>
          </div>
        </div>

        {/* ============ Top-level aggregate cards (from /argo/summary) ============ */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5 mt-4">
          <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="text-[11px] font-mono text-slate-500 uppercase tracking-wider mb-1">
              ARGO Profiles
            </div>
            <div className="text-2xl font-extrabold font-mono text-slate-900">
              {summaryLoading ? '…' : summaryError ? '—' : (summary?.profile_count ?? '—')}
            </div>
            <div className="mt-2 text-[11px] text-slate-500 border-t border-slate-200 pt-1.5 font-mono">
              Offline evaluation set
            </div>
          </div>

          <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="text-[11px] font-mono text-slate-500 uppercase tracking-wider mb-1">
              Unique WMO Floats
            </div>
            <div className="text-2xl font-extrabold font-mono text-slate-900">
              {summaryLoading ? '…' : summaryError ? '—' : (summary?.unique_wmo_count ?? '—')}
            </div>
            <div className="mt-2 text-[11px] text-slate-500 border-t border-slate-200 pt-1.5 font-mono">
              Distinct profiling platforms
            </div>
          </div>

          <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="text-[11px] font-mono text-slate-500 uppercase tracking-wider mb-1">
              Matched Observations
            </div>
            <div className="text-2xl font-extrabold font-mono text-slate-900">
              {summaryLoading
                ? '…'
                : summaryError
                  ? '—'
                  : (summary?.matched_observation_count ?? '—')}
            </div>
            <div className="mt-2 text-[11px] text-slate-500 border-t border-slate-200 pt-1.5 font-mono">
              Profile-depth pairs evaluated
            </div>
          </div>

          <div className="p-4 bg-amber-50/60 border border-amber-200 rounded-lg">
            <div className="text-[11px] font-mono text-amber-800 uppercase tracking-wider mb-1">
              Aggregate RMSE
            </div>
            <div className="text-2xl font-extrabold font-mono text-amber-950">
              {summaryLoading ? '…' : summaryError ? '—' : fmt(summary?.rmse ?? null, 4, ' °C')}
            </div>
            <div className="mt-2 text-[11px] text-amber-800 border-t border-amber-200 pt-1.5 font-mono flex justify-between">
              <span>MAE {fmt(summary?.mae ?? null, 4, ' °C')}</span>
              <span>Bias {signed(summary?.bias ?? null, 4, ' °C')}</span>
            </div>
          </div>
        </div>

        {summaryError && (
          <div className="mt-3 p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-semibold block mb-0.5">Backend unavailable:</span>
              <span>{summaryError}</span>
            </div>
            <button
              type="button"
              onClick={loadSummary}
              className="text-[11px] font-semibold bg-rose-100 hover:bg-rose-200 border border-rose-300 px-2 py-1 rounded shrink-0"
            >
              Retry
            </button>
          </div>
        )}
      </div>

      {/* ============ Interactive workspace: map + selector | comparison ============ */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* -------- Left column: ARGO map + searchable profile selector -------- */}
        <div className="lg:col-span-5 space-y-6">
          <OceanMap
            latitude={selectedProfile?.latitude ?? 15.0}
            longitude={selectedProfile?.longitude ?? 85.0}
            onSelectCoordinates={() => {
              /* ARGO mode never re-targets reconstruction coordinates */
            }}
            argoProfiles={profilesData?.profiles ?? []}
            selectedProfileId={selectedId}
            onSelectProfile={handleSelectProfile}
            title="ARGO Profile Locations"
            subtitle="Click a float marker to load its observational comparison"
            showPresets={false}
            showCoordinateReadout={false}
            showReticle={false}
          />

          {/* -------- Searchable profile selector -------- */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-xs">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-3">
              <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
                <Search className="w-4 h-4 text-sky-700" />
                Select ARGO Profile
              </h3>
              <span className="text-[10px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                {profilesLoading
                  ? 'Loading…'
                  : `${filteredProfiles.length} / ${profilesData?.profiles.length ?? 0}`}
              </span>
            </div>

            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search WMO, date, lat/lon or profile ID…"
              className="w-full text-xs font-mono px-3 py-2 rounded-lg border border-slate-300 bg-white mb-3 focus:outline-hidden focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
            />

            {profilesLoading && (
              <div className="py-8 text-center text-xs text-slate-500 flex flex-col items-center gap-2">
                <RefreshCw className="w-4 h-4 animate-spin text-sky-600" />
                Loading ARGO profiles...
              </div>
            )}

            {!profilesLoading && profilesError && (
              <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <span className="font-semibold block mb-0.5">Backend unavailable:</span>
                  <span>{profilesError}</span>
                </div>
                <button
                  type="button"
                  onClick={() => loadProfiles(true)}
                  className="text-[11px] font-semibold bg-rose-100 hover:bg-rose-200 border border-rose-300 px-2 py-1 rounded shrink-0"
                >
                  Retry
                </button>
              </div>
            )}

            {!profilesLoading && !profilesError && filteredProfiles.length === 0 && (
              <p className="py-6 text-center text-xs text-slate-500">
                No ARGO profile matches “{search}”.
              </p>
            )}

            {!profilesLoading && !profilesError && filteredProfiles.length > 0 && (
              <ul className="max-h-72 overflow-y-auto space-y-1.5 pr-1">
                {filteredProfiles.map((p) => {
                  const isActive = p.profile_id === selectedId;
                  return (
                    <li key={p.profile_id}>
                      <button
                        type="button"
                        onClick={() => handleSelectProfile(p.profile_id)}
                        className={`w-full text-left px-3 py-2 rounded-lg border transition-colors ${
                          isActive
                            ? 'bg-amber-50 border-amber-300 ring-1 ring-amber-300'
                            : 'bg-white border-slate-200 hover:bg-slate-50'
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span
                            className={`text-xs font-bold font-mono ${
                              isActive ? 'text-amber-900' : 'text-slate-800'
                            }`}
                          >
                            WMO {p.wmo}
                          </span>
                          <span className="text-[10px] font-mono text-slate-500">{p.date}</span>
                        </div>
                        <div className="flex items-center justify-between gap-2 mt-0.5 text-[10px] font-mono text-slate-500">
                          <span>
                            {p.latitude.toFixed(2)}°N, {p.longitude.toFixed(2)}°E
                          </span>
                          <span>{p.valid_target_depth_count}/15 depths</span>
                        </div>
                        <div className="text-[9px] font-mono text-slate-400 mt-0.5 truncate">
                          {p.profile_id}
                        </div>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>

        {/* -------- Right column: observed vs predicted comparison -------- */}
        <div className="lg:col-span-7 space-y-6">
          {profilesError && (
            <div className="bg-white border border-rose-200 rounded-xl p-6 shadow-xs text-center">
              <AlertCircle className="w-6 h-6 text-rose-500 mx-auto mb-2" />
              <p className="text-sm font-bold text-rose-800">ARGO comparison unavailable</p>
              <p className="text-xs text-rose-600 mt-1">
                The ARGO evaluation service could not be reached. Start the FastAPI backend and
                retry.
              </p>
            </div>
          )}

          {!profilesError && !selectedId && EMPTY_SELECTION_PLACEHOLDER}

          {!profilesError && selectedId && comparisonLoading && (
            <div className="bg-white border border-slate-200 rounded-xl p-8 shadow-xs flex flex-col items-center justify-center min-h-[300px] text-center">
              <RefreshCw className="w-5 h-5 animate-spin text-sky-600 mb-3" />
              <p className="text-sm font-semibold text-slate-700">Loading comparison...</p>
              <p className="text-xs text-slate-500 mt-1">
                Running OceanEmbed v3 inference server-side for WMO{' '}
                {selectedProfile?.wmo ?? '…'} at {selectedProfile?.date ?? '…'}
              </p>
            </div>
          )}

          {!profilesError && selectedId && !comparisonLoading && comparisonError && (
            <div className="bg-white border border-rose-200 rounded-xl p-6 shadow-xs text-center">
              <AlertCircle className="w-6 h-6 text-rose-500 mx-auto mb-2" />
              <p className="text-sm font-bold text-rose-800">ARGO comparison unavailable</p>
              <p className="text-xs text-rose-600 mt-1 mb-3">{comparisonError}</p>
              <button
                type="button"
                onClick={() => {
                  // Force a re-run of the same comparison request
                  const id = selectedId;
                  setSelectedId(null);
                  setTimeout(() => setSelectedId(id), 0);
                }}
                className="text-xs font-semibold bg-rose-50 hover:bg-rose-100 border border-rose-300 text-rose-800 px-3 py-1.5 rounded"
              >
                Retry comparison
              </button>
            </div>
          )}

          {!profilesError && selectedId && !comparisonLoading && !comparisonError && comparison && (
            <>
              {/* ---- Main scientific visualization ---- */}
              <ArgoComparisonChart comparison={comparison} />

              {/* ---- Per-profile metrics (dynamically computed by backend) ---- */}
              <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-xs">
                <div className="flex items-center justify-between pb-2 border-b border-slate-100 mb-3">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-amber-600" />
                    Per-Profile Metrics
                  </h3>
                  <span className="text-[10px] font-mono text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
                    computed in {fmt(comparison.evaluation_ms, 1, ' ms')}
                  </span>
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
                  <div className="p-2.5 bg-amber-50/60 border border-amber-200 rounded-lg">
                    <span className="text-[10px] font-mono text-amber-800 uppercase block">RMSE</span>
                    <span className="text-lg font-bold font-mono text-amber-950">
                      {fmt(comparison.metrics.rmse, 4)}
                      <span className="text-xs font-normal text-amber-700 ml-0.5">°C</span>
                    </span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[10px] font-mono text-slate-500 uppercase block">MAE</span>
                    <span className="text-lg font-bold font-mono text-slate-900">
                      {fmt(comparison.metrics.mae, 4)}
                      <span className="text-xs font-normal text-slate-500 ml-0.5">°C</span>
                    </span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[10px] font-mono text-slate-500 uppercase block">Bias</span>
                    <span className="text-lg font-bold font-mono text-slate-900">
                      {signed(comparison.metrics.bias, 4)}
                      <span className="text-xs font-normal text-slate-500 ml-0.5">°C</span>
                    </span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[10px] font-mono text-slate-500 uppercase block">
                      Pearson r
                    </span>
                    <span className="text-lg font-bold font-mono text-slate-900">
                      {fmt(comparison.metrics.corr, 4)}
                    </span>
                  </div>
                </div>

                {/* Spatial / temporal matching information */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 mt-3">
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[10px] font-mono text-slate-500 uppercase block">
                      WMO Float
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-900">
                      {comparison.argo_profile.wmo}
                    </span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[10px] font-mono text-slate-500 uppercase block">
                      Observation Date
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-900">
                      {comparison.argo_profile.date}
                    </span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[10px] font-mono text-slate-500 uppercase block">
                      Location
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-900">
                      {comparison.argo_profile.latitude.toFixed(2)}°N,{' '}
                      {comparison.argo_profile.longitude.toFixed(2)}°E
                    </span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[10px] font-mono text-slate-500 uppercase block">
                      Matched Depths
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-900">
                      {comparison.metrics.count} observations
                    </span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[10px] font-mono text-slate-500 uppercase block">
                      Spatial Offset
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-900">
                      {fmt(comparison.spatial_offset_km, 2, ' km')}
                    </span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[10px] font-mono text-slate-500 uppercase block">
                      Temporal Offset
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-900">
                      {fmt(comparison.temporal_offset_hours, 2, ' h')}
                    </span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[10px] font-mono text-slate-500 uppercase block">
                      Matched Grid Cell
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-900">
                      {comparison.grid_latitude.toFixed(2)}°N,{' '}
                      {comparison.grid_longitude.toFixed(2)}°E
                    </span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[10px] font-mono text-slate-500 uppercase block">
                      Matched Model Date
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-900">
                      {comparison.matched_model_date}
                    </span>
                  </div>
                </div>
              </div>

              {/* ---- Depth-wise comparison table (every target depth) ---- */}
              <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-xs">
                <div className="flex items-center justify-between pb-2 border-b border-slate-100 mb-3">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                    <Ruler className="w-3.5 h-3.5 text-sky-700" />
                    Depth-wise Comparison
                  </h3>
                  <span className="text-[10px] font-mono text-slate-500">
                    {comparison.metrics.count} of {comparison.depths_m.length} depths observed
                  </span>
                </div>

                <div className="overflow-x-auto border border-slate-200 rounded-lg">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-slate-50 border-b border-slate-200 text-[11px] font-mono text-slate-600 uppercase">
                      <tr>
                        <th className="py-2 px-3">Depth (m)</th>
                        <th className="py-2 px-3">ARGO Observed (°C)</th>
                        <th className="py-2 px-3">OceanEmbed (°C)</th>
                        <th className="py-2 px-3">Error (°C)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 font-mono">
                      {comparison.depth_comparison.map((row) => {
                        const isThermo = isInThermoclineBand(row.depth_m);
                        return (
                          <tr
                            key={row.depth_m}
                            className={
                              isThermo ? 'bg-amber-50/40 hover:bg-amber-50/70' : 'hover:bg-slate-50'
                            }
                          >
                            <td className="py-1.5 px-3 font-semibold text-slate-800">
                              {row.depth_m}
                              {isThermo && (
                                <span className="text-[9px] font-sans bg-amber-200/80 text-amber-900 px-1 py-0.2 rounded ml-1.5">
                                  thermocline
                                </span>
                              )}
                            </td>
                            <td className="py-1.5 px-3 text-amber-800 font-semibold">
                              {row.observed && row.observed_c !== null
                                ? row.observed_c.toFixed(4)
                                : 'not observed'}
                            </td>
                            <td className="py-1.5 px-3 text-sky-900 font-semibold">
                              {row.predicted_c !== null ? row.predicted_c.toFixed(4) : '—'}
                            </td>
                            <td
                              className={`py-1.5 px-3 font-semibold ${
                                row.error_c === null
                                  ? 'text-slate-400'
                                  : row.error_c > 0
                                    ? 'text-rose-700'
                                    : 'text-emerald-700'
                              }`}
                            >
                              {row.error_c === null
                                ? '—'
                                : `${row.error_c > 0 ? '+' : ''}${row.error_c.toFixed(4)}`}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="text-[10px] text-slate-500 mt-1.5 font-mono">
                  Unobserved depths are shown as “not observed” — values are never imputed in the
                  browser.
                </p>
              </div>

              {/* ---- Observation Gap — the uniqueness story ---- */}
              <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-xs">
                <div className="flex items-center justify-between pb-2 border-b border-slate-100 mb-3">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                    <Database className="w-3.5 h-3.5 text-teal-700" />
                    Observation Gap
                  </h3>
                  <span className="text-[10px] font-mono text-teal-700 bg-teal-50 border border-teal-200 px-1.5 py-0.5 rounded">
                    Evaluation-only
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[11px] text-slate-600 font-semibold block mb-1">
                      Spatial offset
                    </span>
                    <span className="text-lg font-bold font-mono text-slate-900">
                      {fmt(comparison.spatial_offset_km, 2, ' km')}
                    </span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">
                      Float → matched 0.25° grid centre
                    </span>
                  </div>
                  <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                    <span className="text-[11px] text-slate-600 font-semibold block mb-1">
                      Temporal offset
                    </span>
                    <span className="text-lg font-bold font-mono text-slate-900">
                      {fmt(comparison.temporal_offset_hours, 2, ' h')}
                    </span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">
                      Cast time vs 12:00 UTC daily mean
                    </span>
                  </div>
                  <div className="p-3 bg-teal-50/60 border border-teal-200 rounded-lg">
                    <span className="text-[11px] text-teal-800 font-semibold block mb-1">
                      Surface-input completeness
                    </span>
                    <span className="text-lg font-bold font-mono text-teal-950">
                      {completenessPct !== null ? `${completenessPct}%` : '—'}
                    </span>
                    <span className="text-[10px] text-teal-700 block mt-0.5">
                      {comparison.surface_input_availability.available_count} of{' '}
                      {comparison.surface_input_availability.total_count} surface channels present
                    </span>
                  </div>
                </div>

                <div className="mt-3 p-3 bg-teal-50/50 border border-teal-200 rounded-lg text-xs text-teal-950 leading-relaxed">
                  <span className="font-bold block mb-0.5">Why this matters:</span>
                  ARGO provides sparse direct subsurface measurements. OceanEmbed uses surface
                  observations to reconstruct the subsurface profile between direct observations.
                  The model never sees ARGO temperature — the float is the referee, not the input.
                </div>

                {!comparison.is_valid_ocean && (
                  <div className="mt-2 p-2.5 bg-rose-50 border border-rose-200 rounded-lg text-[11px] text-rose-900 flex items-start gap-2">
                    <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                    <span>
                      The matched 0.25° grid cell has <strong>no valid SST observation</strong> on
                      this date, so Reconstruction Mode (<code className="font-mono">/predict</code>)
                      would reject it. The reconstruction shown here was produced from degraded
                      surface input and is kept in the aggregate for transparency; treat its error
                      with caution.
                    </span>
                  </div>
                )}

                {!comparison.surface_input_availability.is_complete && (
                  <div className="mt-2 p-2.5 bg-amber-50 border border-amber-200 rounded-lg text-[11px] text-amber-900 flex items-start gap-2">
                    <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                    <span>
                      Missing surface channels at this cell:{' '}
                      <strong>
                        {comparison.surface_input_availability.missing_channels.join(', ')}
                      </strong>
                      . The backend still evaluates this profile; degraded inputs are reported,
                      never silently dropped.
                    </span>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {/* ================= Aggregate ARGO Validation ================= */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 pb-4 border-b border-slate-200 mb-5">
          <div>
            <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
              Aggregate ARGO Validation
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Every profile in the offline September 2020 evaluation set, reconstructed and
              compared server-side. Values are computed by the backend at request time via{' '}
              <code className="font-mono">GET /argo/summary</code>.
            </p>
          </div>
          <span className="text-[11px] font-mono bg-slate-100 text-slate-600 border border-slate-200 px-2.5 py-1 rounded shrink-0">
            {summary ? `computed in ${fmt(summary.evaluation_ms, 1, ' ms')}` : 'awaiting backend'}
          </span>
        </div>

        {summaryLoading && (
          <div className="py-10 text-center text-xs text-slate-500 flex flex-col items-center gap-2">
            <RefreshCw className="w-4 h-4 animate-spin text-sky-600" />
            Loading aggregate ARGO validation...
          </div>
        )}

        {!summaryLoading && summaryError && (
          <div className="p-4 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-semibold block mb-0.5">ARGO aggregate evaluation unavailable:</span>
              <span>{summaryError}</span>
            </div>
            <button
              type="button"
              onClick={loadSummary}
              className="text-[11px] font-semibold bg-rose-100 hover:bg-rose-200 border border-rose-300 px-2 py-1 rounded shrink-0"
            >
              Retry
            </button>
          </div>
        )}

        {!summaryLoading && !summaryError && summary && (
          <>
            {/* Headline metrics */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-5">
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">Profiles</span>
                <span className="text-xl font-extrabold font-mono text-slate-900">
                  {summary.profile_count}
                </span>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Unique floats
                </span>
                <span className="text-xl font-extrabold font-mono text-slate-900">
                  {summary.unique_wmo_count}
                </span>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Matched obs
                </span>
                <span className="text-xl font-extrabold font-mono text-slate-900">
                  {summary.matched_observation_count}
                </span>
              </div>
              <div className="p-3 bg-amber-50/60 border border-amber-200 rounded-lg">
                <span className="text-[10px] font-mono text-amber-800 uppercase block">RMSE</span>
                <span className="text-xl font-extrabold font-mono text-amber-950">
                  {fmt(summary.rmse, 4, ' °C')}
                </span>
              </div>
              <div className="p-3 bg-amber-50/60 border border-amber-200 rounded-lg">
                <span className="text-[10px] font-mono text-amber-800 uppercase block">MAE</span>
                <span className="text-xl font-extrabold font-mono text-amber-950">
                  {fmt(summary.mae, 4, ' °C')}
                </span>
              </div>
              <div className="p-3 bg-amber-50/60 border border-amber-200 rounded-lg">
                <span className="text-[10px] font-mono text-amber-800 uppercase block">Bias</span>
                <span className="text-xl font-extrabold font-mono text-amber-950">
                  {signed(summary.bias, 4, ' °C')}
                </span>
              </div>
            </div>

            {/* Depth-wise aggregate performance (returned by /argo/summary) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
                    Depth-wise Aggregate RMSE
                  </h3>
                  <div className="flex items-center gap-2 text-[11px] text-slate-500">
                    <span className="w-2.5 h-2.5 rounded-xs bg-amber-100 border border-amber-300 inline-block" />
                    <span>
                      Nominal thermocline band ({THERMOCLINE_TOP_M}–{THERMOCLINE_BOTTOM_M} m)
                    </span>
                  </div>
                </div>
                <div className="overflow-x-auto border border-slate-200 rounded-lg max-h-96 overflow-y-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-slate-50 border-b border-slate-200 text-[11px] font-mono text-slate-600 uppercase sticky top-0">
                      <tr>
                        <th className="py-2 px-3">Depth</th>
                        <th className="py-2 px-3">N</th>
                        <th className="py-2 px-3">RMSE</th>
                        <th className="py-2 px-3">MAE</th>
                        <th className="py-2 px-3">Bias</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 font-mono">
                      {summary.depth_wise.map((row) => {
                        const isThermo = isInThermoclineBand(row.depth_m);
                        return (
                          <tr
                            key={row.depth_m}
                            className={
                              isThermo ? 'bg-amber-50/40 hover:bg-amber-50/70' : 'hover:bg-slate-50'
                            }
                          >
                            <td className="py-1.5 px-3 font-semibold text-slate-800">
                              {row.depth_m} m
                            </td>
                            <td className="py-1.5 px-3 text-slate-600">{row.count}</td>
                            <td className="py-1.5 px-3 font-bold text-sky-800">
                              {row.rmse === null ? '—' : row.rmse.toFixed(4)}
                            </td>
                            <td className="py-1.5 px-3 text-slate-700">
                              {row.mae === null ? '—' : row.mae.toFixed(4)}
                            </td>
                            <td
                              className={`py-1.5 px-3 ${
                                row.bias === null
                                  ? 'text-slate-400'
                                  : row.bias > 0
                                    ? 'text-rose-700'
                                    : 'text-emerald-700'
                              }`}
                            >
                              {row.bias === null
                                ? '—'
                                : `${row.bias > 0 ? '+' : ''}${row.bias.toFixed(4)}`}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="text-[10px] text-slate-500 mt-1.5 font-mono">
                  All values in °C, computed at request time across every evaluated profile.
                </p>
              </div>

              {/* Collocation & provenance metadata */}
              <div className="space-y-3">
                <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg text-xs">
                  <div className="font-bold text-slate-800 mb-1.5">Evaluation Provenance</div>
                  <p className="text-slate-600 leading-relaxed">{summary.source}</p>
                </div>

                <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
                  <div className="font-bold text-slate-800 mb-2 text-xs">
                    Spatial &amp; Temporal Collocation
                  </div>
                  <div className="grid grid-cols-2 gap-2.5 text-xs font-mono">
                    <div className="bg-white border border-slate-200 rounded p-2">
                      <span className="text-[10px] text-slate-500 uppercase block">
                        Mean spatial offset
                      </span>
                      <span className="font-bold text-slate-900">
                        {fmt(summary.profile_metadata.mean_spatial_offset_km, 2, ' km')}
                      </span>
                    </div>
                    <div className="bg-white border border-slate-200 rounded p-2">
                      <span className="text-[10px] text-slate-500 uppercase block">
                        Max spatial offset
                      </span>
                      <span className="font-bold text-slate-900">
                        {fmt(summary.profile_metadata.max_spatial_offset_km, 2, ' km')}
                      </span>
                    </div>
                    <div className="bg-white border border-slate-200 rounded p-2">
                      <span className="text-[10px] text-slate-500 uppercase block">
                        Mean temporal offset
                      </span>
                      <span className="font-bold text-slate-900">
                        {fmt(summary.profile_metadata.mean_temporal_offset_hours, 2, ' h')}
                      </span>
                    </div>
                    <div className="bg-white border border-slate-200 rounded p-2">
                      <span className="text-[10px] text-slate-500 uppercase block">
                        Max temporal offset
                      </span>
                      <span className="font-bold text-slate-900">
                        {fmt(summary.profile_metadata.max_temporal_offset_hours, 2, ' h')}
                      </span>
                    </div>
                  </div>
                  <div className="mt-2.5 text-[11px] text-slate-600 font-mono leading-relaxed">
                    Source files: {summary.profile_metadata.source_files.join(', ')}
                    <br />
                    Observation dates: {summary.profile_metadata.observation_dates.join(', ')}
                    <br />
                    Screening: {summary.profile_metadata.raw_profiles_read} profiles read →{' '}
                    {summary.profile_metadata.profiles_inside_nio_domain} inside domain →{' '}
                    {summary.profile_count} evaluated
                  </div>
                </div>

                {summary.profile_metadata.profiles_with_degraded_surface_inputs.length > 0 && (
                  <div className="p-3.5 bg-amber-50/60 border border-amber-200 rounded-lg text-xs text-amber-950 leading-relaxed">
                    <span className="font-bold block mb-1">
                      Profiles on cells with incomplete surface input (
                      {summary.profile_metadata.profiles_with_degraded_surface_inputs.length}):
                    </span>
                    <ul className="space-y-1 font-mono text-[11px]">
                      {summary.profile_metadata.profiles_with_degraded_surface_inputs.map((p) => (
                        <li key={p.profile_id}>
                          <button
                            type="button"
                            onClick={() => handleSelectProfile(p.profile_id)}
                            className="underline underline-offset-2 hover:text-amber-700"
                          >
                            WMO {p.wmo} ({p.profile_id})
                          </button>
                          : {p.available_count}/7 channels
                          {summary.profile_metadata.profiles_mapped_to_non_ocean_cells.includes(
                            p.profile_id
                          ) && ', no valid SST'}{' '}
                          — missing {p.missing_channels.join(', ')}
                        </li>
                      ))}
                    </ul>
                    <p className="mt-1.5 text-[11px]">
                      {summary.profile_metadata.input_completeness_note}
                    </p>
                  </div>
                )}

                <div className="p-3.5 bg-sky-50/60 border border-sky-200 rounded-lg text-xs text-sky-900 leading-relaxed">
                  <span className="font-bold block mb-0.5">Computation note (from backend):</span>
                  {summary.note}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  );
};
