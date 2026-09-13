import os
import xarray as xr
import numpy as np
import pandas as pd

fpath = "data/argo/argo2020/20200915_prof.nc"
ds = xr.open_dataset(fpath)

print("=" * 80)
print("INSPECTING 20200915_prof.nc")
print("=" * 80)

print(f"Dimensions: {dict(ds.dims)}")
print(f"Variables: {list(ds.variables.keys())}")

lats = ds["LATITUDE"].values
lons = ds["LONGITUDE"].values
times = ds["JULD"].values
wmo = ds["PLATFORM_NUMBER"].values if "PLATFORM_NUMBER" in ds else None
cycles = ds["CYCLE_NUMBER"].values if "CYCLE_NUMBER" in ds else None

nio_mask = (lats >= 5.0) & (lats <= 30.0) & (lons >= 45.0) & (lons <= 105.0)
nio_indices = np.where(nio_mask)[0]

print(f"\nTotal profiles: {len(lats)}")
print(f"NIO profiles: {len(nio_indices)}")

target_depths = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])

for idx in nio_indices:
    pres = ds["PRES"].values[idx]
    temp = ds["TEMP"].values[idx]
    wmo_str = "".join([c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in wmo[idx]]).strip() if wmo is not None else f"float_{idx}"
    cycle_val = int(cycles[idx]) if cycles is not None else -1
    
    # Valid points
    valid = (~np.isnan(pres)) & (~np.isnan(temp)) & (pres >= 0) & (temp > -2.0) & (temp < 40.0)
    p_v = pres[valid]
    t_v = temp[valid]
    
    # QC flags if present
    temp_qc = ds["TEMP_QC"].values[idx] if "TEMP_QC" in ds else None
    
    p_min = float(p_v.min()) if len(p_v) > 0 else None
    p_max = float(p_v.max()) if len(p_v) > 0 else None
    t_min = float(t_v.min()) if len(t_v) > 0 else None
    t_max = float(t_v.max()) if len(t_v) > 0 else None

    print(f"Profile {idx:2d} | Float: {wmo_str:>8s} | Cycle: {cycle_val:3d} | Lat: {lats[idx]:6.2f}N | Lon: {lons[idx]:6.2f}E | Time: {str(times[idx])[:19]} | Pts: {len(p_v):3d} | P_range: [{p_min}, {p_max}] | T_range: [{t_min:.1f}, {t_max:.1f}]")

ds.close()
