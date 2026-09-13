"""
SIH26066 — Real Week 1 (2020-01-01 to 2020-01-07) Acquisition Script
Directly downloads missing real slices from Copernicus Marine API
with strict NetCDF timestamp verification. Zero pilot copying.
"""

import os
import sys
import time

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import xarray as xr
import copernicusmarine as cm

from src.data.catalog import TARGET_GRID, DATA_CATALOG
from src.data.acquisition import CopernicusAcquisitionEngine

def verify_file_dates(filepath, required_dates, required_vars):
    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"
    try:
        with xr.open_dataset(filepath) as ds:
            for v in required_vars:
                if v not in ds.variables and v not in ds.data_vars:
                    return False, f"Missing variable {v}"
            times = [str(t)[:10] for t in ds.time.values]
            for d in required_dates:
                if d not in times:
                    return False, f"Date {d} not in dataset timestamps ({times[:3]}..{times[-3:]})"
            return True, f"Verified {len(required_dates)} dates in {filepath} ({os.path.getsize(filepath):,} bytes)"
    except Exception as e:
        return False, str(e)

def download_winds_week1():
    out_dir = "data/raw/winds"
    os.makedirs(out_dir, exist_ok=True)
    out_file = "winds_2020-01-01_2020-01-07.nc"
    out_path = os.path.join(out_dir, out_file)
    
    spec = DATA_CATALOG["WINDS"]
    req_dates = [f"2020-01-0{d}" for d in range(1, 8)]
    
    ok, msg = verify_file_dates(out_path, req_dates, spec.variables)
    if ok:
        print(f"[WINDS] Already verified: {msg}")
        return out_path
        
    print(f"[WINDS] Downloading {spec.dataset_id} for 2020-01-01 to 2020-01-07...")
    t0 = time.time()
    cm.subset(
        dataset_id=spec.dataset_id,
        variables=spec.variables,
        minimum_longitude=TARGET_GRID.lon_min - 0.2,
        maximum_longitude=TARGET_GRID.lon_max + 0.2,
        minimum_latitude=TARGET_GRID.lat_min - 0.2,
        maximum_latitude=TARGET_GRID.lat_max + 0.2,
        start_datetime="2020-01-01T00:00:00",
        end_datetime="2020-01-07T23:59:59",
        output_directory=out_dir,
        output_filename=out_file,
        overwrite=True
    )
    ok, msg = verify_file_dates(out_path, req_dates, spec.variables)
    if not ok:
        raise RuntimeError(f"WINDS download verification failed: {msg}")
    print(f"[WINDS] Success in {time.time()-t0:.2f}s: {msg}")
    return out_path

def download_glorys_day(target_date, engine=None):
    if engine is None:
        engine = CopernicusAcquisitionEngine()
    
    # Check if this day is already covered by an existing valid cache file
    cached = engine.find_cached_slice("GLORYS", target_date, target_date)
    if cached:
        print(f"[GLORYS {target_date}] Already present in verified cache: {cached} ({os.path.getsize(cached):,} bytes)")
        return cached

    out_dir = "data/raw/glorys"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"glorys_{target_date}_{target_date}.nc"
    out_path = os.path.join(out_dir, out_file)
    
    spec = DATA_CATALOG["GLORYS"]
    req_dates = [target_date]
    
    ok, msg = verify_file_dates(out_path, req_dates, spec.variables)
    if ok:
        print(f"[GLORYS {target_date}] Already verified: {msg}")
        return out_path

    print(f"[GLORYS {target_date}] Downloading {spec.dataset_id}...")
    t0 = time.time()
    cm.subset(
        dataset_id=spec.dataset_id,
        variables=spec.variables,
        minimum_longitude=TARGET_GRID.lon_min - 0.2,
        maximum_longitude=TARGET_GRID.lon_max + 0.2,
        minimum_latitude=TARGET_GRID.lat_min - 0.2,
        maximum_latitude=TARGET_GRID.lat_max + 0.2,
        minimum_depth=0.49,
        maximum_depth=1063.0,
        start_datetime=f"{target_date}T00:00:00",
        end_datetime=f"{target_date}T23:59:59",
        output_directory=out_dir,
        output_filename=out_file,
        overwrite=True
    )
    ok, msg = verify_file_dates(out_path, req_dates, spec.variables)
    if not ok:
        raise RuntimeError(f"GLORYS {target_date} verification failed: {msg}")
    print(f"[GLORYS {target_date}] Success in {time.time()-t0:.2f}s: {msg}")
    return out_path

def audit_week1():
    engine = CopernicusAcquisitionEngine()
    week1_dates = [f"2020-01-0{d}" for d in range(1, 8)]
    datasets = ["SST", "SSH", "CURRENTS", "SSS", "WINDS", "GLORYS"]
    
    print("\n" + "=" * 70)
    print("WEEK 1 (2020-01-01 to 2020-01-07) COMPREHENSIVE DATASET AUDIT")
    print("=" * 70)
    
    all_ok = True
    for d in week1_dates:
        row_status = []
        for ds_key in datasets:
            found = engine.find_cached_slice(ds_key, d, d)
            if found:
                row_status.append(f"{ds_key}: OK")
            else:
                row_status.append(f"{ds_key}: MISSING")
                all_ok = False
        print(f"[{d}] " + " | ".join(row_status))
    
    print("=" * 70)
    if all_ok:
        print("STATUS: ALL 7 DAYS x 6 DATASETS VERIFIED IN LOCAL CACHE!")
    else:
        print("STATUS: INCOMPLETE — Some datasets or dates are missing.")
    print("=" * 70)
    return all_ok

def main():
    print("=" * 70)
    print("SIH26066 — REAL DATA ACQUISITION: 2020-01-01 to 2020-01-07")
    print("=" * 70)
    
    engine = CopernicusAcquisitionEngine()

    # 1. WINDS
    download_winds_week1()
    
    # 2. GLORYS Days 1 through 7
    week1_dates = [f"2020-01-0{d}" for d in range(1, 8)]
    for d_str in week1_dates:
        download_glorys_day(d_str, engine=engine)

    # 3. Final Audit
    audit_week1()

if __name__ == "__main__":
    main()

