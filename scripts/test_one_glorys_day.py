"""
SIH26066 — Isolated Single Day GLORYS Benchmark Test
Tests acquisition of 2020-02-01 GLORYS 3D thetao with strict finite timeout.
Measures request duration, file size, download speed, and validates NetCDF contents.
"""

import os
import sys
import time
import json
from datetime import datetime
import concurrent.futures
import xarray as xr
import copernicusmarine as cm

sys.path.insert(0, os.path.abspath("."))
from src.data.catalog import TARGET_GRID, DATA_CATALOG

def run_single_glorys_download(target_date: str = "2020-02-01", timeout_seconds: int = 180) -> dict:
    spec = DATA_CATALOG["GLORYS"]
    out_dir = "data/raw/glorys"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"glorys_{target_date}_{target_date}.nc"
    out_path = os.path.join(out_dir, out_file)

    test_result = {
        "target_date": target_date,
        "dataset_id": spec.dataset_id,
        "variable": "thetao",
        "spatial_bounds": {
            "lon_min": TARGET_GRID.lon_min - 0.2,
            "lon_max": TARGET_GRID.lon_max + 0.2,
            "lat_min": TARGET_GRID.lat_min - 0.2,
            "lat_max": TARGET_GRID.lat_max + 0.2,
        },
        "depth_bounds": {"min_depth": 0.49, "max_depth": 1063.0},
        "timeout_seconds": timeout_seconds,
        "request_start": datetime.utcnow().isoformat() + "Z",
        "status": "UNKNOWN",
        "wall_clock_duration_seconds": None,
        "file_size_bytes": None,
        "file_size_mb": None,
        "verification": {}
    }

    print("=" * 70)
    print(f"TESTING ONE GLORYS DAY ACQUISITION: {target_date}")
    print(f"Dataset:   {spec.dataset_id}")
    print(f"Variable:  thetao")
    print(f"Region:    {TARGET_GRID.lat_min-0.2}°N to {TARGET_GRID.lat_max+0.2}°N, {TARGET_GRID.lon_min-0.2}°E to {TARGET_GRID.lon_max+0.2}°E")
    print(f"Depths:    0.49m to 1063.0m")
    print(f"Timeout:   {timeout_seconds}s")
    print("=" * 70)

    t0 = time.time()

    def do_download():
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(do_download)
        try:
            res = future.result(timeout=timeout_seconds)
            t_elapsed = time.time() - t0
            test_result["wall_clock_duration_seconds"] = round(t_elapsed, 2)
            test_result["request_end"] = datetime.utcnow().isoformat() + "Z"
            print(f"\nAPI Call returned in {t_elapsed:.2f} seconds.")

            # Validate the file
            if not os.path.exists(out_path):
                test_result["status"] = "FAILURE"
                test_result["error"] = f"Output file does not exist: {out_path}"
                return test_result

            fsize = os.path.getsize(out_path)
            test_result["file_size_bytes"] = fsize
            test_result["file_size_mb"] = round(fsize / (1024 * 1024), 2)

            with xr.open_dataset(out_path) as ds:
                times = [str(t)[:10] for t in ds.time.values]
                depths = ds.depth.values.tolist()
                lats = ds.latitude.values.tolist()
                lons = ds.longitude.values.tolist()

                test_result["verification"] = {
                    "variables": list(ds.data_vars.keys()),
                    "timestamps": times,
                    "num_depth_levels": len(depths),
                    "depth_range": [round(float(depths[0]), 2), round(float(depths[-1]), 2)],
                    "spatial_shape": [len(lats), len(lons)],
                    "date_verified": target_date in times,
                    "thetao_mean": float(ds["thetao"].mean().item()),
                    "thetao_valid_cells": int((~ds["thetao"].isel(time=0, depth=0).isnull()).sum().item())
                }

            if test_result["verification"]["date_verified"] and "thetao" in test_result["verification"]["variables"]:
                test_result["status"] = "SUCCESS"
                print(f"VERIFICATION SUCCESS: {out_path} ({test_result['file_size_mb']} MB)")
                print(f"  Depths: {test_result['verification']['num_depth_levels']} levels ({test_result['verification']['depth_range'][0]}m to {test_result['verification']['depth_range'][1]}m)")
                print(f"  Shape:  {test_result['verification']['spatial_shape']}")
                print(f"  Mean:   {test_result['verification']['thetao_mean']:.2f}°C")
            else:
                test_result["status"] = "FAILURE"
                test_result["error"] = "File verification checks failed."

        except concurrent.futures.TimeoutError:
            t_elapsed = time.time() - t0
            test_result["wall_clock_duration_seconds"] = round(t_elapsed, 2)
            test_result["request_end"] = datetime.utcnow().isoformat() + "Z"
            test_result["status"] = "TIMEOUT"
            test_result["error"] = f"Request timed out after {timeout_seconds} seconds."
            print(f"\nFAILURE: Request TIMED OUT after {timeout_seconds} seconds!")

        except Exception as e:
            t_elapsed = time.time() - t0
            test_result["wall_clock_duration_seconds"] = round(t_elapsed, 2)
            test_result["request_end"] = datetime.utcnow().isoformat() + "Z"
            test_result["status"] = "FAILURE"
            test_result["error"] = str(e)
            print(f"\nFAILURE: Error during download: {e}")

    return test_result

if __name__ == "__main__":
    os.makedirs("reports/real", exist_ok=True)
    res = run_single_glorys_download("2020-02-01", timeout_seconds=180)
    out_json = "reports/real/glorys_one_day_test.json"
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved test report to {out_json}")
    print(f"FINAL RESULT: {res['status']}")
    if res["status"] != "SUCCESS":
        sys.exit(1)
