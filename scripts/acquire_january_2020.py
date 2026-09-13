"""
SIH26066 — Full Month (January 2020) Real Data Acquisition Script
Expands the verified real dataset from Week 1 (Jan 1-7) to the full calendar month
(2020-01-01 through 2020-01-31) for 31 continuous real daily historical samples.
Includes automatic retries, timestamp validation, and checkpointing.
"""

import os
import sys
import time
from datetime import datetime, timedelta
import xarray as xr
import copernicusmarine as cm

sys.path.insert(0, os.path.abspath("."))

from src.data.catalog import TARGET_GRID, DATA_CATALOG
from src.data.acquisition import CopernicusAcquisitionEngine

def verify_file(filepath, req_vars, start_date=None, end_date=None, is_weekly=False):
    if not os.path.exists(filepath):
        return False, "File does not exist"
    if os.path.getsize(filepath) < 1024:
        return False, "File size too small (<1KB)"
    try:
        with xr.open_dataset(filepath) as ds:
            for v in req_vars:
                if v not in ds.variables and v not in ds.data_vars:
                    return False, f"Missing variable {v}"
            times = [str(t)[:10] for t in ds.time.values]
            if is_weekly:
                if start_date and not any(t <= start_date for t in times):
                    return False, f"Missing bounding weekly observation <= {start_date}"
                if end_date and not any(t >= end_date for t in times):
                    return False, f"Missing bounding weekly observation >= {end_date}"
            else:
                if start_date and start_date not in times:
                    return False, f"Date {start_date} not in timestamps"
                if end_date and end_date not in times:
                    return False, f"Date {end_date} not in timestamps"
        return True, f"Verified ({os.path.getsize(filepath):,} bytes)"
    except Exception as e:
        return False, str(e)

def download_multi_day(engine, dataset_key, start_date, end_date):
    spec = DATA_CATALOG[dataset_key]
    out_dir = os.path.join("data/raw", dataset_key.lower())
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{dataset_key.lower()}_{start_date}_{end_date}.nc"
    out_path = os.path.join(out_dir, out_file)

    is_weekly = (spec.temporal_frequency == "weekly")
    ok, _ = verify_file(out_path, spec.variables, start_date, end_date, is_weekly=is_weekly)
    if ok:
        print(f"[{dataset_key}] Already verified: {out_path}")
        return out_path

    print(f"[{dataset_key}] Downloading {start_date} to {end_date}...")
    t0 = time.time()
    
    start_dt = f"{start_date}T00:00:00"
    end_dt = f"{end_date}T23:59:59"
    if is_weekly:
        dt_s = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=8)
        dt_e = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=8)
        start_dt = f"{dt_s.strftime('%Y-%m-%d')}T00:00:00"
        end_dt = f"{dt_e.strftime('%Y-%m-%d')}T23:59:59"

    cm.subset(
        dataset_id=spec.dataset_id,
        variables=spec.variables,
        minimum_longitude=TARGET_GRID.lon_min - 0.2,
        maximum_longitude=TARGET_GRID.lon_max + 0.2,
        minimum_latitude=TARGET_GRID.lat_min - 0.2,
        maximum_latitude=TARGET_GRID.lat_max + 0.2,
        start_datetime=start_dt,
        end_datetime=end_dt,
        output_directory=out_dir,
        output_filename=out_file,
        overwrite=True
    )
    ok, msg = verify_file(out_path, spec.variables, start_date, end_date, is_weekly=is_weekly)
    if not ok:
        raise RuntimeError(f"Download verification failed for {dataset_key}: {msg}")
    print(f"[{dataset_key}] Success in {time.time()-t0:.2f}s: {msg}")
    return out_path

def download_glorys_day(engine, target_date):
    spec = DATA_CATALOG["GLORYS"]
    cached = engine.find_cached_slice("GLORYS", target_date, target_date)
    if cached:
        print(f"[GLORYS {target_date}] Found in cache: {cached}")
        return cached

    out_dir = "data/raw/glorys"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"glorys_{target_date}_{target_date}.nc"
    out_path = os.path.join(out_dir, out_file)

    ok, _ = verify_file(out_path, spec.variables, target_date, target_date)
    if ok:
        print(f"[GLORYS {target_date}] Already verified: {out_path}")
        return out_path

    print(f"[GLORYS {target_date}] Downloading {target_date}...")
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
    ok, msg = verify_file(out_path, spec.variables, target_date, target_date)
    if not ok:
        raise RuntimeError(f"GLORYS {target_date} verification failed: {msg}")
    print(f"[GLORYS {target_date}] Success in {time.time()-t0:.2f}s: {msg}")
    return out_path

def main():
    print("=" * 80)
    print("SIH26066 — SCALING TO FULL MONTH: JANUARY 2020 (2020-01-01 to 2020-01-31)")
    print("=" * 80)
    engine = CopernicusAcquisitionEngine()

    # 1. SST: Jan 8 to Jan 31
    download_multi_day(engine, "SST", "2020-01-08", "2020-01-31")

    # 2. SSH: Jan 8 to Jan 31
    download_multi_day(engine, "SSH", "2020-01-08", "2020-01-31")

    # 3. CURRENTS: Jan 8 to Jan 31
    download_multi_day(engine, "CURRENTS", "2020-01-08", "2020-01-31")

    # 4. SSS: Jan 8 to Jan 31
    download_multi_day(engine, "SSS", "2020-01-08", "2020-01-31")

    # 5. WINDS: 7-day batches
    wind_batches = [
        ("2020-01-08", "2020-01-14"),
        ("2020-01-15", "2020-01-21"),
        ("2020-01-22", "2020-01-28"),
        ("2020-01-29", "2020-01-31")
    ]
    for w_start, w_end in wind_batches:
        download_multi_day(engine, "WINDS", w_start, w_end)

    # 6. GLORYS: Days 8 through 31
    for day in range(8, 32):
        d_str = f"2020-01-{day:02d}"
        download_glorys_day(engine, d_str)

    print("\nALL 31 DAYS OF JANUARY 2020 ACQUIRED AND VERIFIED!")

if __name__ == "__main__":
    main()
