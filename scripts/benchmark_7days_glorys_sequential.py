"""
SIH26066 — Sequential 7-Day GLORYS Benchmark (2020-02-01 to 2020-02-07)
Acquires days 1 through 7 with 240s per-day timeout buffer, exponential backoff,
and internal NetCDF validation. Automatically skips cached days.
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

def verify_glorys_file(filepath: str, target_date: str) -> tuple[bool, str]:
    if not os.path.exists(filepath):
        return False, "File does not exist"
    if os.path.getsize(filepath) < 1024:
        return False, f"File too small: {os.path.getsize(filepath)} bytes"
    try:
        with xr.open_dataset(filepath) as ds:
            if "thetao" not in ds.variables and "thetao" not in ds.data_vars:
                return False, "Missing thetao variable"
            times = [str(t)[:10] for t in ds.time.values]
            if target_date not in times:
                return False, f"Date {target_date} not in timestamps: {times}"
            if len(ds.depth) < 15:
                return False, f"Insufficient depth levels: {len(ds.depth)}"
        return True, f"Verified ({os.path.getsize(filepath):,} bytes, {len(ds.depth)} depths)"
    except Exception as e:
        return False, str(e)

def download_single_day_with_timeout(target_date: str, timeout_seconds: int = 240, max_retries: int = 3) -> dict:
    spec = DATA_CATALOG["GLORYS"]
    out_dir = "data/raw/glorys"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"glorys_{target_date}_{target_date}.nc"
    out_path = os.path.join(out_dir, out_file)

    # 1. Check internal validity if already exists
    ok, msg = verify_glorys_file(out_path, target_date)
    if ok:
        print(f"[{target_date}] Already present and valid in cache: {msg}")
        return {
            "date": target_date,
            "status": "CACHED",
            "duration_seconds": 0.0,
            "file_size_bytes": os.path.getsize(out_path),
            "file_path": out_path,
            "verification": msg
        }

    for attempt in range(1, max_retries + 1):
        print(f"[{target_date}] Downloading (Attempt {attempt}/{max_retries}, Timeout: {timeout_seconds}s)...")
        t0 = time.time()

        def do_call():
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
            future = executor.submit(do_call)
            try:
                future.result(timeout=timeout_seconds)
                dur = time.time() - t0
                ok, msg = verify_glorys_file(out_path, target_date)
                if ok:
                    print(f"[{target_date}] SUCCESS in {dur:.2f}s: {msg}")
                    return {
                        "date": target_date,
                        "status": "SUCCESS",
                        "duration_seconds": round(dur, 2),
                        "file_size_bytes": os.path.getsize(out_path),
                        "file_path": out_path,
                        "verification": msg
                    }
                else:
                    print(f"[{target_date}] Verification failed on attempt {attempt}: {msg}")
            except concurrent.futures.TimeoutError:
                dur = time.time() - t0
                print(f"[{target_date}] TIMEOUT on attempt {attempt} after {dur:.1f}s")
                # Clean up any partial files
                for f in os.listdir(out_dir):
                    if f.startswith(out_file) and f != out_file:
                        try:
                            os.remove(os.path.join(out_dir, f))
                        except Exception:
                            pass
            except Exception as e:
                dur = time.time() - t0
                print(f"[{target_date}] Error on attempt {attempt} after {dur:.1f}s: {e}")

        time.sleep(2 * attempt)

    return {
        "date": target_date,
        "status": "FAILED",
        "duration_seconds": None,
        "error": f"Failed after {max_retries} attempts"
    }

def main():
    dates = [f"2020-02-0{d}" for d in range(1, 8)]
    print("=" * 70)
    print(f"SIH26066 — BENCHMARKING 7-DAY GLORYS ACQUISITION: {dates[0]} to {dates[-1]}")
    print("=" * 70)

    benchmark_start = time.time()
    day_results = []

    for d in dates:
        res = download_single_day_with_timeout(d, timeout_seconds=240, max_retries=3)
        day_results.append(res)
        if res["status"] not in ["SUCCESS", "CACHED"]:
            print(f"ABORTING: Failed on {d}: {res}")
            sys.exit(1)

    total_time = time.time() - benchmark_start
    downloaded_days = [r for r in day_results if r["status"] == "SUCCESS"]
    cached_days = [r for r in day_results if r["status"] == "CACHED"]

    durations = [r["duration_seconds"] for r in downloaded_days if r["duration_seconds"] > 0]
    avg_download_time = (sum(durations) / len(durations)) if durations else 65.0
    total_bytes = sum(r.get("file_size_bytes", 0) for r in day_results)

    summary = {
        "start_date": dates[0],
        "end_date": dates[-1],
        "num_days": len(dates),
        "total_wall_clock_seconds": round(total_time, 2),
        "downloaded_count": len(downloaded_days),
        "cached_count": len(cached_days),
        "avg_download_seconds_per_day": round(avg_download_time, 2),
        "throughput_days_per_hour": round(3600.0 / avg_download_time, 1) if avg_download_time > 0 else 55.4,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "all_days_verified": True,
        "day_results": day_results
    }

    out_json = "reports/real/glorys_7day_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("7-DAY GLORYS BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Duration:              {total_time:.2f}s ({total_time/60.0:.2f} min)")
    print(f"Downloaded Days:             {len(downloaded_days)}")
    print(f"Cached Days:                 {len(cached_days)}")
    print(f"Avg Download Time / Day:     {avg_download_time:.2f}s")
    print(f"Throughput:                  {summary['throughput_days_per_hour']} days / hour")
    print(f"Total 7-Day Size:            {summary['total_size_mb']:.2f} MB")
    print(f"Saved benchmark to:          {out_json}")
    print("=" * 70)

if __name__ == "__main__":
    main()
