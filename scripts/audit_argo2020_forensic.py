import os
import glob
import json
import xarray as xr
import numpy as np
import pandas as pd

print("=" * 80)
print("PHASE D — ARGO 2020 DATA FORENSIC VALIDATION")
print("=" * 80)

nc_files = sorted(glob.glob("data/argo/argo2020/*.nc"))
print(f"Found {len(nc_files)} NetCDF files in data/argo/argo2020/: {nc_files}")

lat_min, lat_max = 5.0, 30.0
lon_min, lon_max = 45.0, 105.0

all_profiles = []
total_raw_profiles = 0

for fpath in nc_files:
    fname = os.path.basename(fpath)
    ds = xr.open_dataset(fpath)
    lats = ds["LATITUDE"].values
    lons = ds["LONGITUDE"].values
    times = ds["JULD"].values
    wmo = ds["PLATFORM_NUMBER"].values if "PLATFORM_NUMBER" in ds else None
    cycles = ds["CYCLE_NUMBER"].values if "CYCLE_NUMBER" in ds else None
    
    n_prof = len(lats)
    total_raw_profiles += n_prof
    
    for idx in range(n_prof):
        lat = float(lats[idx])
        lon = float(lons[idx])
        time_val = times[idx]
        
        # Check if inside NIO domain
        in_nio = (lat >= lat_min) and (lat <= lat_max) and (lon >= lon_min) and (lon <= lon_max)
        
        # WMO float ID string decode
        if wmo is not None:
            wmo_chars = wmo[idx]
            wmo_str = "".join([c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in wmo_chars]).strip()
        else:
            wmo_str = f"UNKNOWN_{idx}"
            
        cycle_val = int(cycles[idx]) if cycles is not None else -1
        
        pres = ds["PRES"].values[idx]
        temp = ds["TEMP"].values[idx]
        pres_qc = ds["PRES_QC"].values[idx] if "PRES_QC" in ds else None
        temp_qc = ds["TEMP_QC"].values[idx] if "TEMP_QC" in ds else None
        
        # Valid physical values: In tropical North Indian Ocean, water column temp is strictly >= 2.0 degC
        # Dummy uncalibrated/failed sensor transmissions filled with 0.0 are excluded
        valid = (~np.isnan(pres)) & (~np.isnan(temp)) & (pres >= 0) & (temp >= 2.0) & (temp < 40.0)
        p_v = pres[valid]
        t_v = temp[valid]
        
        # Check if profile has enough valid points and a physical surface temperature
        is_physically_authentic = len(t_v) >= 10 and np.max(t_v) >= 15.0
        
        # Decode QC flags if present
        # ARGO QC: 1=good, 2=probably good, 3=probably bad, 4=bad
        good_qc_count = 0
        if temp_qc is not None:
            t_qc_v = temp_qc[valid]
            # Convert bytes to char if needed
            t_qc_chars = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in t_qc_v]
            good_qc_count = sum(1 for c in t_qc_chars if c in ['1', '2'])
            
        p_min = float(p_v.min()) if len(p_v) > 0 else None
        p_max = float(p_v.max()) if len(p_v) > 0 else None
        t_min = float(t_v.min()) if len(t_v) > 0 else None
        t_max = float(t_v.max()) if len(t_v) > 0 else None
        
        all_profiles.append({
            "source_file": fname,
            "profile_idx_in_file": idx,
            "wmo_float_id": wmo_str,
            "cycle_number": cycle_val,
            "latitude": lat,
            "longitude": lon,
            "timestamp": str(time_val)[:19],
            "in_nio": in_nio,
            "raw_levels_count": len(pres),
            "valid_pts": len(p_v),
            "p_min_dbar": p_min,
            "p_max_dbar": p_max,
            "t_min_degC": t_min,
            "t_max_degC": t_max,
            "reaches_100m": bool(p_max >= 100.0) if p_max is not None else False,
            "reaches_200m": bool(p_max >= 200.0) if p_max is not None else False,
            "reaches_500m": bool(p_max >= 500.0) if p_max is not None else False,
            "reaches_700m": bool(p_max >= 700.0) if p_max is not None else False,
            "reaches_1000m": bool(p_max >= 1000.0) if p_max is not None else False,
            "good_qc_fraction": round(good_qc_count / len(p_v), 3) if len(p_v) > 0 and temp_qc is not None else 1.0,
            "is_physically_authentic": bool(is_physically_authentic)
        })
    ds.close()

# Filter NIO profiles
nio_profiles = [p for p in all_profiles if p["in_nio"] and p["is_physically_authentic"]]
unique_wmo = sorted(list(set(p["wmo_float_id"] for p in nio_profiles)))

# Date range
actual_dates = sorted([p["timestamp"] for p in nio_profiles])
min_date = actual_dates[0] if actual_dates else None
max_date = actual_dates[-1] if actual_dates else None

# Depths reached
reaches_100 = sum(1 for p in nio_profiles if p["reaches_100m"])
reaches_200 = sum(1 for p in nio_profiles if p["reaches_200m"])
reaches_500 = sum(1 for p in nio_profiles if p["reaches_500m"])
reaches_700 = sum(1 for p in nio_profiles if p["reaches_700m"])
reaches_1000 = sum(1 for p in nio_profiles if p["reaches_1000m"])

lats_nio = [p["latitude"] for p in nio_profiles]
lons_nio = [p["longitude"] for p in nio_profiles]
t_mins = [p["t_min_degC"] for p in nio_profiles if p["t_min_degC"] is not None]
t_maxs = [p["t_max_degC"] for p in nio_profiles if p["t_max_degC"] is not None]

forensic_report = {
    "audit_timestamp": "2026-09-13T15:15:00Z",
    "files_audited": nc_files,
    "total_raw_profiles_in_files": total_raw_profiles,
    "nio_profiles_count": len(nio_profiles),
    "unique_wmo_floats_count": len(unique_wmo),
    "unique_wmo_floats": unique_wmo,
    "actual_date_range": [min_date, max_date],
    "spatial_coverage": {
        "lat_min": round(min(lats_nio), 4) if lats_nio else None,
        "lat_max": round(max(lats_nio), 4) if lats_nio else None,
        "lon_min": round(min(lons_nio), 4) if lons_nio else None,
        "lon_max": round(max(lons_nio), 4) if lons_nio else None
    },
    "temperature_range_degC": [round(min(t_mins), 2) if t_mins else None, round(max(t_maxs), 2) if t_maxs else None],
    "depth_reach_summary": {
        "reaches_100m": reaches_100,
        "reaches_200m": reaches_200,
        "reaches_500m": reaches_500,
        "reaches_700m": reaches_700,
        "reaches_1000m": reaches_1000
    },
    "profiles": nio_profiles
}

os.makedirs("reports/argo2020", exist_ok=True)
out_json = "reports/argo2020/argo2020_forensic.json"
with open(out_json, "w") as f:
    json.dump(forensic_report, f, indent=2)

print(f"Successfully generated: {out_json}")
print(f"Total NIO Profiles: {len(nio_profiles)}")
print(f"Unique Floats: {len(unique_wmo)}")
print(f"Date range: {min_date} to {max_date}")
print(f"Lats: [{min(lats_nio):.2f}, {max(lats_nio):.2f}], Lons: [{min(lons_nio):.2f}, {max(lons_nio):.2f}]")
print(f"Reaching 1000m: {reaches_1000} / {len(nio_profiles)}")
