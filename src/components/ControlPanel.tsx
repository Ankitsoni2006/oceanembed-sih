import React, { useEffect, useState } from 'react';
import { Calendar, Play, AlertCircle, RefreshCw, SlidersHorizontal, Info } from 'lucide-react';
import { AvailableDatesResponse } from '../types';

interface ControlPanelProps {
  latitude: number;
  longitude: number;
  onChangeLatitude: (lat: number) => void;
  onChangeLongitude: (lon: number) => void;
  date: string;
  onChangeDate: (date: string) => void;
  availableDates: AvailableDatesResponse | null;
  isDatesLoading: boolean;
  onReconstruct: () => void;
  isLoading: boolean;
  error: string | null;
  onClearError: () => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  latitude,
  longitude,
  onChangeLatitude,
  onChangeLongitude,
  date,
  onChangeDate,
  availableDates,
  isDatesLoading,
  onReconstruct,
  isLoading,
  error,
  onClearError,
}) => {
  // Keep the raw text the user types so a cleared or partially typed field is not
  // silently replaced by a default value; only finite numbers are sent upward.
  const [latText, setLatText] = useState<string>(String(latitude));
  const [lonText, setLonText] = useState<string>(String(longitude));

  // Sync when coordinates change elsewhere (map click, preset chip)
  useEffect(() => {
    if (parseFloat(latText) !== latitude) setLatText(String(latitude));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latitude]);
  useEffect(() => {
    if (parseFloat(lonText) !== longitude) setLonText(String(longitude));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [longitude]);

  const latNum = parseFloat(latText);
  const lonNum = parseFloat(lonText);
  const isLatValid = Number.isFinite(latNum) && latNum >= 5.0 && latNum <= 30.0;
  const isLonValid = Number.isFinite(lonNum) && lonNum >= 45.0 && lonNum <= 105.0;
  const canSubmit = isLatValid && isLonValid && !isLoading;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-xs flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between pb-3 border-b border-slate-200 mb-4">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-sky-700" />
            Observation Controls
          </h2>
          <span className="text-[11px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
            Available Data
          </span>
        </div>

        {/* Coordinate Input Fields */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Latitude (°N)
            </label>
            <div className="relative">
              <input
                type="number"
                step="0.25"
                min="5.0"
                max="30.0"
                value={latText}
                onChange={(e) => {
                  onClearError();
                  setLatText(e.target.value);
                  const v = parseFloat(e.target.value);
                  if (Number.isFinite(v)) onChangeLatitude(v);
                }}
                disabled={isLoading}
                className={`w-full text-sm font-mono px-3 py-2 rounded-lg border bg-white focus:outline-hidden focus:ring-2 ${
                  isLatValid
                    ? 'border-slate-300 focus:ring-sky-500 focus:border-sky-500'
                    : 'border-rose-400 bg-rose-50 text-rose-800 focus:ring-rose-500'
                }`}
              />
              <span className="absolute right-3 top-2.5 text-xs text-slate-400 font-mono">
                °N
              </span>
            </div>
            <span className="text-[10px] text-slate-500 mt-1 block">
              Range: 5.00° to 30.00°
            </span>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Longitude (°E)
            </label>
            <div className="relative">
              <input
                type="number"
                step="0.25"
                min="45.0"
                max="105.0"
                value={lonText}
                onChange={(e) => {
                  onClearError();
                  setLonText(e.target.value);
                  const v = parseFloat(e.target.value);
                  if (Number.isFinite(v)) onChangeLongitude(v);
                }}
                disabled={isLoading}
                className={`w-full text-sm font-mono px-3 py-2 rounded-lg border bg-white focus:outline-hidden focus:ring-2 ${
                  isLonValid
                    ? 'border-slate-300 focus:ring-sky-500 focus:border-sky-500'
                    : 'border-rose-400 bg-rose-50 text-rose-800 focus:ring-rose-500'
                }`}
              />
              <span className="absolute right-3 top-2.5 text-xs text-slate-400 font-mono">
                °E
              </span>
            </div>
            <span className="text-[10px] text-slate-500 mt-1 block">
              Range: 45.00° to 105.00°
            </span>
          </div>
        </div>

        {/* Date Selector */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1">
            <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-slate-500" />
              Observation Snapshot Date
            </label>
            <span className="text-[10px] text-slate-500 font-mono">
              {availableDates ? `${availableDates.total_dates} daily fields` : 'Loading dates...'}
            </span>
          </div>

          <div className="relative">
            <select
              value={date}
              onChange={(e) => {
                onClearError();
                onChangeDate(e.target.value);
              }}
              disabled={isLoading || isDatesLoading}
              className="w-full text-sm font-mono px-3 py-2 rounded-lg border border-slate-300 bg-white focus:outline-hidden focus:ring-2 focus:ring-sky-500 focus:border-sky-500 cursor-pointer"
            >
              {availableDates && availableDates.dates.length > 0 ? (
                availableDates.dates.map((dt) => (
                  <option key={dt} value={dt}>
                    {dt} {dt === '2020-09-15' ? '— (Default Held-Out Test)' : ''}
                  </option>
                ))
              ) : (
                <option value="2020-09-15">2020-09-15 (Default Benchmark)</option>
              )}
            </select>
          </div>
          <p className="text-[11px] text-slate-500 mt-1.5 leading-tight flex items-start gap-1">
            <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
            <span>
              Processed (archived) satellite surface inputs are available daily for Jan 1 – Sep 30,
              2020. Sep 2020 is the held-out test month.
            </span>
          </p>
        </div>
      </div>

      {/* Action Button & Error Alert */}
      <div>
        {error && (
          <div className="mb-3 p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800 flex items-start gap-2 leading-relaxed">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-semibold block mb-0.5">Observation Notice:</span>
              <span>{error}</span>
            </div>
          </div>
        )}

        <button
          type="button"
          onClick={onReconstruct}
          disabled={!canSubmit}
          className={`w-full py-2.5 px-4 rounded-lg font-semibold text-sm flex items-center justify-center gap-2 shadow-xs transition-colors ${
            canSubmit
              ? 'bg-sky-800 hover:bg-sky-900 text-white cursor-pointer active:scale-[0.99]'
              : 'bg-slate-200 text-slate-400 cursor-not-allowed'
          }`}
        >
          {isLoading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin text-white" />
              <span>Running OceanEmbed inference...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>Reconstruct Temperature</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
