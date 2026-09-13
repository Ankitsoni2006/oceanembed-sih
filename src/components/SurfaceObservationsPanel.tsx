import React from 'react';
import { Satellite, Wind, Droplets, Waves, Gauge } from 'lucide-react';
import { SurfaceObservations } from '../types';

interface SurfaceObservationsPanelProps {
  observations?: SurfaceObservations | null;
  date: string;
}

export const SurfaceObservationsPanel: React.FC<SurfaceObservationsPanelProps> = ({
  observations,
  date,
}) => {
  if (!observations) {
    return null;
  }

  const items = [
    {
      label: 'Sea Surface Temp (SST)',
      val: observations.sst_c !== null ? `${observations.sst_c.toFixed(2)} °C` : '—',
      icon: Droplets,
      desc: 'Surface thermal state',
    },
    {
      label: 'Sea Surface Salinity (SSS)',
      val: observations.sss_psu !== null ? `${observations.sss_psu.toFixed(2)} PSU` : '—',
      icon: Gauge,
      desc: 'Surface halocline state',
    },
    {
      label: 'Sea Surface Height (SSH/SLA)',
      val: observations.ssh_m !== null ? `${observations.ssh_m.toFixed(3)} m` : '—',
      icon: Waves,
      desc: 'Altimetric height anomaly',
    },
    {
      label: 'Surface Current (U East)',
      val: observations.u_current_ms !== null ? `${observations.u_current_ms.toFixed(2)} m/s` : '—',
      icon: Waves,
      desc: 'Zonal surface velocity',
    },
    {
      label: 'Surface Current (V North)',
      val: observations.v_current_ms !== null ? `${observations.v_current_ms.toFixed(2)} m/s` : '—',
      icon: Waves,
      desc: 'Meridional surface velocity',
    },
    {
      label: '10m Wind Velocity (U East)',
      val: observations.u_wind_ms !== null ? `${observations.u_wind_ms.toFixed(2)} m/s` : '—',
      icon: Wind,
      desc: '10m zonal wind vector',
    },
    {
      label: '10m Wind Velocity (V North)',
      val: observations.v_wind_ms !== null ? `${observations.v_wind_ms.toFixed(2)} m/s` : '—',
      icon: Wind,
      desc: '10m meridional wind vector',
    },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-xs">
      <div className="flex items-center justify-between pb-3 border-b border-slate-200 mb-3">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
            <Satellite className="w-4 h-4 text-sky-700" />
            Surface Satellite Observations
          </h3>
          <p className="text-[11px] text-slate-500">
            Real satellite surface inputs extracted at target grid cell ({date})
          </p>
        </div>
        <span className="text-[10px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
          7 Input Variables
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2.5">
        {items.map((it) => {
          const Icon = it.icon;
          return (
            <div key={it.label} className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
              <div className="flex items-center gap-1.5 text-slate-500 mb-1">
                <Icon className="w-3.5 h-3.5 text-sky-700" />
                <span className="text-[10px] font-mono uppercase tracking-wider truncate">
                  {it.label}
                </span>
              </div>
              <div className="text-sm font-bold font-mono text-slate-900">{it.val}</div>
              <div className="text-[9px] text-slate-400 mt-0.5 truncate">{it.desc}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
