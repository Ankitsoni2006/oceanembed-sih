import React from 'react';
import { Activity, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';
import { HealthResponse } from '../types';
import { api } from '../services/api';

interface HeaderProps {
  health: HealthResponse | null;
  isHealthLoading: boolean;
}

export const Header: React.FC<HeaderProps> = ({ health, isHealthLoading }) => {
  const isOnline = health && health.status === 'ok';

  return (
    <header className="border-b border-slate-200 bg-white sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Left: Brand & Subtitle */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-sky-800 flex items-center justify-center text-white font-semibold shadow-sm">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg tracking-tight text-slate-900">
                OceanEmbed
              </span>
            </div>
            <p className="text-xs text-slate-500 font-normal">
              Subsurface Temperature Reconstruction
            </p>
          </div>
        </div>

        {/* Right: Live Backend Status */}
        <div className="flex items-center gap-3">

          {/* Real Backend Status Indicator */}
          <div
            className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${
              isHealthLoading
                ? 'bg-slate-50 text-slate-500 border-slate-200'
                : isOnline
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                : 'bg-rose-50 text-rose-700 border-rose-200'
            }`}
            title={
              isOnline
                ? `FastAPI connected at ${api.baseUrl} (${health.device.toUpperCase()})`
                : `FastAPI backend is not reachable at ${api.baseUrl}`
            }
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isHealthLoading
                  ? 'bg-slate-400 animate-pulse'
                  : isOnline
                  ? 'bg-emerald-500'
                  : 'bg-rose-500'
              }`}
            />
            <span>
              {isHealthLoading
                ? 'Checking Backend...'
                : isOnline
                ? `Backend Connected (${health.device.toUpperCase()})`
                : 'Backend Offline'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
