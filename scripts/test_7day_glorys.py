"""
SIH26066 — 7-Day Multi-Day GLORYS Benchmark Test
Tests whether acquiring a 7-day chunk (2020-02-01 to 2020-02-07) in a single API call
is faster and more reliable than 7 independent daily API calls.
"""

import os
import sys
import time
import json
from datetime import datetime, UTC
import concurrent.futures
import xarray as xr
import copernicusmarine as cm

sys.path.insert(0, os.path.abspath("."))
from src.data.catalog import TARGET_GRID, DATA_CATALOG

def run_7day_glorys_test(start_date: str = "2020-02-01", end_date: str = "2020-02-07", timeout_seconds: int = 300) -> dict:
    spec = DATA_CATALOG["GLORYS"]
    out_dir = "data/raw/glorys"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"glorys_{start_date}_{end_date}.nc"
    out_path = os.path.join(out_dir, out_file)

    test_result = {
        "start_date": start_date,
        "end_date": end_date,
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
        "request_start": datetime.now(UTC).isoformat(),
        "status": "UNKNOWN",
        "wall_clock_duration_seconds": None,
        "file_size_bytes": None,
        "file_size_mb": None,
        "verification": {}
    }

    print("=" * 70)
    print(f"TESTING 7-DAY MULTI-DAY GLORYS ACQUISITION: {start_date} to {end_date}")
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
            start_datetime=f"{start_date}T00:00:00",
            end_datetime=f"{end_date}T23:59:59",
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
            test_result["request_end"] = datetime.now(UTC).isoformat()
            print(f"\nAPI Call returned in {t_elapsed:.2f} seconds.")

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

                expected_dates = [f"2020-02-0{d}" for d in range(1, 8)]
                all_present = all(d in times for d in expected_dates)

                test_result["verification"] = {
                    "variables": list(ds.data_vars.keys()),
                    "timestamps": times,
                    "num_timestamps": len(times),
                    "expected_dates": expected_dates,
                    "all_expected_dates_present": all_present,
                    "num_depth_levels": len(depths),
                    "depth_range": [round(float(depths[0]), 2), round(float(depths[-1]), 2)],
                    "spatial_shape": [len(lats), len(lons)],
                    "thetao_mean": float(ds["thetao"].mean().item()),
                }

            if all_present and "thetao" in test_result["verification"]["variables"]:
                test_result["status"] = "SUCCESS"
                print(f"VERIFICATION SUCCESS: {out_path} ({test_result['file_size_mb']} MB)")
                print(f"  Timestamps: {len(times)} days ({times[0]} to {times[-1]})")
                print(f"  Depths:     {test_result['verification']['num_depth_levels']} levels ({test_result['verification']['depth_range'][0]}m to {test_result['verification']['depth_range'][1]}m)")
                print(f"  Shape:      (time={len(times)}, depth={len(depths)}, lat={len(lats)}, lon={len(lons)})")
                print(f"  Mean Temp:  {test_result['verification']['thetao_mean']:.2f}°C")
                print(f"  Per-Day Equivalent Time: {t_elapsed / 7.0:.2f} s/day")
            else:
                test_result["status"] = "FAILURE"
                test_result["error"] = f"Missing timestamps: expected {expected_dates}, got {times}"

        except concurrent.futures.TimeoutError:
            t_elapsed = time.time() - t0
            test_result["wall_clock_duration_seconds"] = round(t_elapsed, 2)
            test_result["request_end"] = datetime.now(UTC).isoformat()
            test_result["status"] = "TIMEOUT"
            test_result["error"] = f"Request timed out after {timeout_seconds} seconds."
            print(f"\nFAILURE: Request TIMED OUT after {timeout_seconds} seconds!")

        except Exception as e:
            t_elapsed = time.time() - t0
            test_result["wall_clock_duration_seconds"] = round(t_elapsed, 2)
            test_result["request_end"] = datetime.now(UTC).isoformat()
            test_result["status"] = "FAILURE"
            test_result["error"] = str(e)
            print(f"\nFAILURE: Error during download: {e}")

    return test_result

if __name__ == "__main__":
    os.makedirs("reports/real", exist_ok=True)
    res = run_7day_glorys_test("2020-02-01", "2020-02-07", timeout_seconds=300)
    out_json = "reports/real/glorys_7day_test.json"
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved 7-day test report to {out_json}")
    print(f"FINAL RESULT: {res['status']}")
    if res["status"] != "SUCCESS":
        sys.exit(1)
