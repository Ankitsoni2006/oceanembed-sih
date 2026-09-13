import os
import glob
import xarray as xr
import pandas as pd
import numpy as np

print("=" * 80)
print("PHASE B — FORENSIC SEARCH FOR LOCAL ARGO DATA")
print("=" * 80)

extensions = ["*.nc", "*.nc4", "*.csv", "*.json", "*.zip", "*.gz"]
all_files = []
for ext in extensions:
    all_files.extend(glob.glob(f"**/{ext}", recursive=True))

print(f"Total candidate files found across workspace: {len(all_files)}")

argo_candidates = []

for fpath in all_files:
    # Skip non-data or obvious build/node_modules/git directories
    norm_path = fpath.replace("\\", "/")
    if any(skip in norm_path for skip in [".git", "node_modules", ".gemini", "venv", ".idea", "__pycache__"]):
        continue

    # Check NetCDF files
    if norm_path.endswith((".nc", ".nc4")):
        try:
            ds = xr.open_dataset(fpath)
            vars_set = set(ds.variables.keys())
            # ARGO datasets typically have JULD, PRES, TEMP, LATITUDE, LONGITUDE
            is_argo = ("JULD" in vars_set) or ("PRES" in vars_set and "TEMP" in vars_set and "LATITUDE" in vars_set)
            if is_argo or "argo" in norm_path.lower():
                time_info = "No time"
                min_time, max_time = None, None
                if "JULD" in ds:
                    juld = ds["JULD"].values
                    # Remove NaT if any
                    valid_juld = juld[~pd.isna(juld)]
                    if len(valid_juld) > 0:
                        min_time = str(np.min(valid_juld))[:19]
                        max_time = str(np.max(valid_juld))[:19]
                        time_info = f"{min_time} to {max_time}"
                elif "time" in ds:
                    t = ds["time"].values
                    valid_t = t[~pd.isna(t)]
                    if len(valid_t) > 0:
                        min_time = str(np.min(valid_t))[:19]
                        max_time = str(np.max(valid_t))[:19]
                        time_info = f"{min_time} to {max_time}"

                lats = ds["LATITUDE"].values if "LATITUDE" in ds else (ds["lat"].values if "lat" in ds else None)
                lons = ds["LONGITUDE"].values if "LONGITUDE" in ds else (ds["lon"].values if "lon" in ds else None)
                lat_str = f"[{np.nanmin(lats):.2f}, {np.nanmax(lats):.2f}]" if lats is not None else "N/A"
                lon_str = f"[{np.nanmin(lons):.2f}, {np.nanmax(lons):.2f}]" if lons is not None else "N/A"

                n_prof = ds.dims.get("N_PROF", len(lats) if lats is not None else "N/A")

                argo_candidates.append({
                    "file": norm_path,
                    "type": "NetCDF",
                    "n_profiles": n_prof,
                    "date_range": time_info,
                    "min_date": min_time,
                    "max_date": max_time,
                    "lat_range": lat_str,
                    "lon_range": lon_str,
                    "has_temp": "TEMP" in vars_set or "thetao" in vars_set,
                    "has_pres": "PRES" in vars_set or "depth" in vars_set
                })
            ds.close()
        except Exception as e:
            pass

    # Check CSV / JSON files with argo in name
    elif "argo" in norm_path.lower() and norm_path.endswith((".csv", ".json")):
        argo_candidates.append({
            "file": norm_path,
            "type": "CSV/JSON",
            "size_bytes": os.path.getsize(fpath)
        })

print(f"\nDiscovered {len(argo_candidates)} ARGO candidate files:")
for c in argo_candidates:
    print(c)
