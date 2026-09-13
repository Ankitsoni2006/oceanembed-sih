"""
SIH26066 — Audit All Local Dataset Files
Inspects every file under data/pilot, data/raw, data/processed, and data/argo.
Extracts metadata, dimensions, timestamps, variables, and determines provenance.
"""

import os
import json
import xarray as xr
import torch
import numpy as np

def audit_file(filepath):
    stat = os.stat(filepath)
    size = stat.st_size
    basename = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    
    info = {
        'filepath': filepath.replace('\\', '/'),
        'filename': basename,
        'size_bytes': size,
        'type': ext
    }
    
    if ext == '.nc':
        try:
            with xr.open_dataset(filepath) as ds:
                info['attrs_title'] = ds.attrs.get('title', 'N/A')
                info['attrs_id'] = ds.attrs.get('id', ds.attrs.get('dataset_id', 'N/A'))
                info['variables'] = list(ds.data_vars.keys())
                
                if 'time' in ds.coords or 'time' in ds.dims:
                    times = [str(t)[:19] for t in ds['time'].values]
                    info['time_count'] = len(times)
                    info['time_min'] = times[0] if times else None
                    info['time_max'] = times[-1] if times else None
                    info['all_dates'] = sorted(list(set([t[:10] for t in times])))
                else:
                    info['time_count'] = 0
                    info['time_min'] = None
                    info['time_max'] = None
                    info['all_dates'] = []
                
                lat_var = 'lat' if 'lat' in ds.coords else ('latitude' if 'latitude' in ds.coords else None)
                lon_var = 'lon' if 'lon' in ds.coords else ('longitude' if 'longitude' in ds.coords else None)
                
                if lat_var and lon_var:
                    lats = ds[lat_var].values
                    lons = ds[lon_var].values
                    info['lat_range'] = [float(np.min(lats)), float(np.max(lats))]
                    info['lon_range'] = [float(np.min(lons)), float(np.max(lons))]
                    info['lat_count'] = len(lats)
                    info['lon_count'] = len(lons)
                    info['lat_res'] = round(float(abs(lats[1] - lats[0])), 4) if len(lats) > 1 else 'single'
                else:
                    info['lat_range'] = None
                    info['lon_range'] = None
                    info['lat_res'] = None

                depth_var = 'depth' if 'depth' in ds.coords else ('PRES' if 'PRES' in ds.variables else None)
                if depth_var:
                    if depth_var == 'depth':
                        d_vals = ds['depth'].values
                        info['depth_levels'] = [round(float(d), 2) for d in d_vals] if d_vals.ndim == 1 else 'multi-dim'
                    else:
                        info['depth_levels'] = 'ARGO PRES'
                else:
                    info['depth_levels'] = None
        except Exception as e:
            info['error'] = str(e)
            
    elif ext == '.pt':
        try:
            pt = torch.load(filepath, map_location='cpu', weights_only=False)
            info['pt_keys'] = list(pt.keys())
            if 'X' in pt:
                info['X_shape'] = list(pt['X'].shape)
            if 'Y' in pt:
                info['Y_shape'] = list(pt['Y'].shape)
            if 'dates' in pt:
                info['dates'] = pt['dates']
        except Exception as e:
            info['error'] = str(e)
            
    return info

def main():
    results = []
    for folder in ['data/pilot', 'data/raw', 'data/processed', 'data/argo']:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    results.append(audit_file(fp))

    os.makedirs('reports', exist_ok=True)
    with open('reports/full_data_audit.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Audited {len(results)} files. Saved to reports/full_data_audit.json.\n")
    print(f"{'FILEPATH':<45} | {'SIZE':>12} | {'TIME RANGE':<30} | {'VARS':<20} | {'RES':<6} | {'DEPTHS'}")
    print("-" * 130)
    for r in results:
        fpath = r['filepath']
        sz = f"{r['size_bytes']:,} B"
        t_count = r.get('time_count', 0)
        t_min = r.get('time_min', 'N/A')
        t_max = r.get('time_max', 'N/A')
        t_span = f"{t_min[:10]}..{t_max[:10]} ({t_count}t)" if t_count else "No time"
        
        v_list = r.get('variables', []) or r.get('pt_keys', [])
        v_str = ",".join(v_list[:3]) + (f"(+{len(v_list)-3})" if len(v_list) > 3 else "")
        res = str(r.get('lat_res', 'N/A'))
        d = r.get('depth_levels')
        d_str = f"{len(d)} lvls" if isinstance(d, list) else (str(d) if d else "1 (sfc)")
        
        print(f"{fpath:<45} | {sz:>12} | {t_span:<30} | {v_str:<20} | {res:<6} | {d_str}")

if __name__ == '__main__':
    main()
