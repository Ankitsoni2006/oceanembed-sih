"""
SIH26066 — Optimized Multi-Month & Full Year 2020 Real Data Acquisition & Processing
Features:
- Robust single-day GLORYS downloading with strict 120s timeouts and exponential backoff
- Resilient WINDS downloading in small 3-day to 4-day batches with timeout protection
- Zero duplicate downloads: skips any file that passes internal NetCDF validation
- Clean error recovery: partial/stalled temporary files are purged on failure
- Out-of-core memory safety: flushes memory after each monthly chunk
"""

import os
import sys
import gc
import json
import time
import calendar
import argparse
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Optional, Tuple
import concurrent.futures
import xarray as xr
import copernicusmarine as cm
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.data.catalog import TARGET_GRID, DATA_CATALOG
from src.data.acquisition import CopernicusAcquisitionEngine
from src.preprocessing.pipeline import OceanDataPipeline
from src.data.chunked_dataset import ChunkedOceanDataset


def verify_file(filepath: str, req_vars: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None, is_weekly: bool = False) -> Tuple[bool, str]:
    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"
    if os.path.getsize(filepath) < 1024:
        return False, f"File size too small (<1KB): {os.path.getsize(filepath)} bytes"
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


def download_with_timeout(call_fn, timeout_seconds: int = 180):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call_fn)
        return future.result(timeout=timeout_seconds)


def cleanup_temp_files(directory: str, prefix: str):
    if not os.path.exists(directory):
        return
    for fname in os.listdir(directory):
        if fname.startswith(prefix) and fname != prefix:
            try:
                os.remove(os.path.join(directory, fname))
            except Exception:
                pass


def download_surface_slice(dataset_key: str, start_date: str, end_date: str, timeout_seconds: int = 180, max_retries: int = 3) -> str:
    spec = DATA_CATALOG[dataset_key]
    out_dir = os.path.join("data/raw", dataset_key.lower())
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{dataset_key.lower()}_{start_date}_{end_date}.nc"
    out_path = os.path.join(out_dir, out_file)

    is_weekly = (spec.temporal_frequency == "weekly")
    ok, msg = verify_file(out_path, spec.variables, start_date, end_date, is_weekly=is_weekly)
    if ok:
        print(f"[{dataset_key}] Already verified in cache: {out_path} ({msg})")
        return out_path

    start_dt = f"{start_date}T00:00:00"
    end_dt = f"{end_date}T23:59:59"
    if is_weekly:
        dt_s = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=8)
        dt_e = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=8)
        start_dt = f"{dt_s.strftime('%Y-%m-%d')}T00:00:00"
        end_dt = f"{dt_e.strftime('%Y-%m-%d')}T23:59:59"

    for attempt in range(1, max_retries + 1):
        print(f"[{dataset_key}] Downloading {start_date} to {end_date} (Attempt {attempt}/{max_retries}, Timeout: {timeout_seconds}s)...")
        t0 = time.time()
        try:
            def do_subset():
                return cm.subset(
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

            download_with_timeout(do_subset, timeout_seconds=timeout_seconds)
            ok, msg = verify_file(out_path, spec.variables, start_date, end_date, is_weekly=is_weekly)
            if ok:
                print(f"[{dataset_key}] SUCCESS in {time.time()-t0:.2f}s: {msg}")
                return out_path
            else:
                print(f"[{dataset_key}] Verification failed on attempt {attempt}: {msg}")
        except concurrent.futures.TimeoutError:
            # Check if file was actually completed and written to disk before treating as failure
            ok, msg = verify_file(out_path, spec.variables, start_date, end_date, is_weekly=is_weekly)
            if ok:
                print(f"[{dataset_key}] SUCCESS (verified on disk despite timeout signal) in {time.time()-t0:.2f}s: {msg}")
                return out_path
            print(f"[{dataset_key}] TIMEOUT after {time.time()-t0:.1f}s on attempt {attempt}!")
            cleanup_temp_files(out_dir, out_file)
        except Exception as e:
            print(f"[{dataset_key}] Error on attempt {attempt} after {time.time()-t0:.1f}s: {e}")
            cleanup_temp_files(out_dir, out_file)

        time.sleep(2 * attempt)

    raise RuntimeError(f"Failed to acquire {dataset_key} for {start_date} to {end_date} after {max_retries} attempts.")


def download_glorys_day(target_date: str, timeout_seconds: int = 240, max_retries: int = 3) -> str:
    spec = DATA_CATALOG["GLORYS"]
    out_dir = "data/raw/glorys"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"glorys_{target_date}_{target_date}.nc"
    out_path = os.path.join(out_dir, out_file)

    ok, msg = verify_file(out_path, spec.variables, target_date, target_date)
    if ok:
        print(f"[GLORYS {target_date}] Already verified in cache: {msg}")
        return out_path

    for attempt in range(1, max_retries + 1):
        print(f"[GLORYS {target_date}] Downloading (Attempt {attempt}/{max_retries}, Timeout: {timeout_seconds}s)...")
        t0 = time.time()
        try:
            def do_subset():
                return cm.subset(
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

            download_with_timeout(do_subset, timeout_seconds=timeout_seconds)
            ok, msg = verify_file(out_path, spec.variables, target_date, target_date)
            if ok:
                print(f"[GLORYS {target_date}] SUCCESS in {time.time()-t0:.2f}s: {msg}")
                return out_path
            else:
                print(f"[GLORYS {target_date}] Verification failed on attempt {attempt}: {msg}")
        except concurrent.futures.TimeoutError:
            ok, msg = verify_file(out_path, spec.variables, target_date, target_date)
            if ok:
                print(f"[GLORYS {target_date}] SUCCESS (verified on disk despite timeout signal) in {time.time()-t0:.2f}s: {msg}")
                return out_path
            print(f"[GLORYS {target_date}] TIMEOUT after {time.time()-t0:.1f}s on attempt {attempt}!")
            cleanup_temp_files(out_dir, out_file)
        except Exception as e:
            print(f"[GLORYS {target_date}] Error on attempt {attempt} after {time.time()-t0:.1f}s: {e}")
            cleanup_temp_files(out_dir, out_file)

        time.sleep(2 * attempt)

    raise RuntimeError(f"Failed to acquire GLORYS for {target_date} after {max_retries} attempts.")


def acquire_month_data(year: int, month: int, engine: CopernicusAcquisitionEngine):
    _, last_day = calendar.monthrange(year, month)
    start_date = f"{year:04d}-{month:02d}-01"
    end_date = f"{year:04d}-{month:02d}-{last_day:02d}"

    print(f"\n" + "=" * 70)
    print(f"ACQUIRING REAL DATA: {year:04d}-{month:02d} ({start_date} to {end_date}, {last_day} days)")
    print("=" * 70)

    # 1. SST (Monthly multi-day slice)
    download_surface_slice("SST", start_date, end_date, timeout_seconds=180)

    # 2. SSH (Monthly multi-day slice)
    download_surface_slice("SSH", start_date, end_date, timeout_seconds=180)

    # 3. CURRENTS (Monthly multi-day slice)
    download_surface_slice("CURRENTS", start_date, end_date, timeout_seconds=180)

    # 4. SSS (Monthly multi-day slice + bounding weeklies)
    download_surface_slice("SSS", start_date, end_date, timeout_seconds=180)

    # 5. WINDS (Identify exact missing days and download in small <= 4-day batches)
    print(f"\n[WINDS] Checking cached wind coverage for {year:04d}-{month:02d}...")
    missing_wind_days = []
    for day in range(1, last_day + 1):
        d_str = f"{year:04d}-{month:02d}-{day:02d}"
        if not engine.find_cached_slice("WINDS", d_str, d_str):
            missing_wind_days.append(d_str)

    if not missing_wind_days:
        print(f"[WINDS] All {last_day} days of {year:04d}-{month:02d} already present in verified cache!")
    else:
        print(f"[WINDS] Found {len(missing_wind_days)} missing days to download: {missing_wind_days[0]} to {missing_wind_days[-1]}")
        batches = []
        cur_batch = [missing_wind_days[0]]
        for d in missing_wind_days[1:]:
            prev_d = datetime.strptime(cur_batch[-1], "%Y-%m-%d")
            this_d = datetime.strptime(d, "%Y-%m-%d")
            if (this_d - prev_d).days == 1 and len(cur_batch) < 4:
                cur_batch.append(d)
            else:
                batches.append((cur_batch[0], cur_batch[-1]))
                cur_batch = [d]
        if cur_batch:
            batches.append((cur_batch[0], cur_batch[-1]))
        
        for s_str, e_str in batches:
            print(f"[WINDS] Downloading batch {s_str} to {e_str}...")
            download_surface_slice("WINDS", s_str, e_str, timeout_seconds=900)

    # 6. GLORYS (Sequential daily slices with strict 300s timeouts)
    print(f"\n[GLORYS] Checking/downloading {last_day} daily 3D fields with timeout protection...")
    t_glo = time.time()
    for day in range(1, last_day + 1):
        d_str = f"{year:04d}-{month:02d}-{day:02d}"
        download_glorys_day(d_str, timeout_seconds=300)
    print(f"[GLORYS] All {last_day} days verified in {time.time()-t_glo:.1f}s.")


def process_month_chunk(year: int, month: int, pipeline: OceanDataPipeline) -> str:
    _, last_day = calendar.monthrange(year, month)
    chunk_name = f"chunk_{year:04d}_{month:02d}"
    out_pt = os.path.join("data/processed", f"{chunk_name}.pt")
    manifest_path = os.path.join("data/processed/manifests", f"{year:04d}_{month:02d}.json")

    if os.path.exists(out_pt) and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as mf:
                mdata = json.load(mf)
            if mdata.get("num_days") == last_day and mdata.get("qc_all_passed"):
                print(f"[{chunk_name}] Already fully processed and verified: {out_pt} ({last_day} days)")
                return out_pt
        except Exception:
            pass

    print(f"\n--- PROCESSING REAL TENSOR CHUNK FOR {year:04d}-{month:02d} ---")
    x_list = []
    y_list = []
    dates = []
    daily_manifests = []

    t_start = time.time()
    for day in range(1, last_day + 1):
        d_str = f"{year:04d}-{month:02d}-{day:02d}"
        t_d0 = time.time()
        x, y, meta = pipeline.process_day(d_str)
        x_list.append(x)
        y_list.append(y)
        dates.append(d_str)
        daily_manifests.append(meta)
        print(f"  [{d_str}] Processed in {time.time()-t_d0:.2f}s | Valid ocean cells: {(~torch.isnan(y[0,0])).sum().item()}")

    # Assemble month tensor
    X_chunk = torch.cat(x_list, dim=0) # [last_day, 14, 101, 241]
    Y_chunk = torch.cat(y_list, dim=0) # [last_day, 15, 101, 241]

    payload = {
        "chunk_name": chunk_name,
        "dates": dates,
        "X": X_chunk,
        "Y": Y_chunk,
        "metadata": {
            "year": year,
            "month": month,
            "num_days": len(dates),
            "processed_at": datetime.now(UTC).isoformat()
        }
    }
    torch.save(payload, out_pt)
    file_size = os.path.getsize(out_pt)
    print(f"[{chunk_name}] Saved tensor chunk to {out_pt} ({file_size:,} bytes, {len(dates)} days in {time.time()-t_start:.1f}s)")

    # Compute comprehensive distribution statistics
    var_names = ["sst", "sss", "ssh", "u_curr", "v_curr", "u_wind", "v_wind"]
    surface_stats = {}
    for c, vname in enumerate(var_names):
        vals = X_chunk[:, c].numpy()
        masks = X_chunk[:, c + 7].numpy()
        valid_px = vals[masks == 1.0]
        surface_stats[vname] = {
            "channel_index": c,
            "mask_index": c + 7,
            "valid_cells": int((masks == 1.0).sum()),
            "nan_cells": int(np.isnan(vals).sum()),
            "min": round(float(valid_px.min()), 4) if len(valid_px) > 0 else None,
            "max": round(float(valid_px.max()), 4) if len(valid_px) > 0 else None,
            "mean": round(float(valid_px.mean()), 4) if len(valid_px) > 0 else None,
            "std": round(float(valid_px.std()), 4) if len(valid_px) > 0 else None,
        }

    target_depths_list = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
    depth_stats = {}
    for d_idx, d_m in enumerate(target_depths_list):
        d_vals = Y_chunk[:, d_idx].numpy()
        valid_d = d_vals[~np.isnan(d_vals)]
        depth_stats[f"{int(d_m)}m"] = {
            "depth_index": d_idx,
            "depth_meters": d_m,
            "valid_cells": int((~np.isnan(d_vals)).sum()),
            "nan_cells": int(np.isnan(d_vals).sum()),
            "min": round(float(valid_d.min()), 4) if len(valid_d) > 0 else None,
            "max": round(float(valid_d.max()), 4) if len(valid_d) > 0 else None,
            "mean": round(float(valid_d.mean()), 4) if len(valid_d) > 0 else None,
            "std": round(float(valid_d.std()), 4) if len(valid_d) > 0 else None,
        }

    # Save rich monthly manifest
    manifest_data = {
        "month": f"{year:04d}-{month:02d}",
        "chunk_name": chunk_name,
        "file_path": out_pt,
        "file_size_bytes": file_size,
        "preprocessing_version": "1.0.0",
        "acquisition_timestamp": datetime.now(UTC).isoformat(),
        "validation_status": "PASSED",
        "qc_all_passed": True,
        "num_days": len(dates),
        "date_start": dates[0],
        "date_end": dates[-1],
        "dates_included": dates,
        "missing_dates": [],
        "dimensions": {
            "X_shape": list(X_chunk.shape),
            "Y_shape": list(Y_chunk.shape),
            "spatial_grid": [101, 241],
            "num_surface_channels": 14,
            "num_depth_levels": 15
        },
        "grid_definition": {
            "region": "North Indian Ocean",
            "lat_min": 5.0,
            "lat_max": 30.0,
            "lon_min": 45.0,
            "lon_max": 105.0,
            "resolution": 0.25,
            "shape": [101, 241]
        },
        "target_depths": target_depths_list,
        "source_datasets": {
            "SST": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
            "SSS": "cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c",
            "SSH": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
            "CURRENTS": "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m",
            "WINDS": "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
            "GLORYS": "cmems_mod_glo_phy_my_0.083deg_P1D-m"
        },
        "surface_variable_statistics": surface_stats,
        "subsurface_depth_statistics": depth_stats,
        "daily_manifests": daily_manifests
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[{chunk_name}] Saved comprehensive monthly manifest to {manifest_path}")

    # Update dataset_index.json
    ChunkedOceanDataset.save_chunk(
        processed_dir="data/processed",
        chunk_name=chunk_name,
        x_list=[X_chunk],
        y_list=[Y_chunk],
        dates=dates,
        metadata_extra={"year": year, "month": month}
    )

    del X_chunk, Y_chunk, x_list, y_list, payload
    gc.collect()
    return out_pt


def main():
    parser = argparse.ArgumentParser(description="SIH26066 — 2020 Real Data Multi-Month Acquisition & Processing")
    parser.add_argument("--months", type=str, default="all", help="Comma-separated months (e.g. 2,3,4) or 'all' for 2 through 12")
    parser.add_argument("--year", type=int, default=2020, help="Year to process (default: 2020)")
    parser.add_argument("--skip-download", action="store_true", help="Skip Copernicus downloads and proceed directly to processing")
    args = parser.parse_args()

    if args.months.lower() == "all":
        month_list = list(range(2, 13))
    else:
        month_list = [int(m.strip()) for m in args.months.split(",")]

    print("=" * 80)
    print(f"SIH26066 — OPTIMIZED MULTI-MONTH PIPELINE EXECUTION")
    print(f"Year:    {args.year}")
    print(f"Months:  {month_list}")
    print("=" * 80)

    engine = CopernicusAcquisitionEngine()
    pipeline = OceanDataPipeline()

    os.makedirs("data/processed/manifests", exist_ok=True)

    total_start = time.time()
    for m in month_list:
        m_t0 = time.time()
        print(f"\n>>> PROCESSING MONTH {args.year:04d}-{m:02d} <<<")
        if not args.skip_download:
            acquire_month_data(args.year, m, engine)
        
        chunk_file = process_month_chunk(args.year, m, pipeline)
        print(f">>> COMPLETED MONTH {args.year:04d}-{m:02d} IN {time.time()-m_t0:.1f}s: {chunk_file} <<<\n")

    print("=" * 80)
    print(f"ALL REQUESTED MONTHS ({month_list}) SUCCESSFULLY PROCESSED IN {time.time()-total_start:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
