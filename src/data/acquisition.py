"""
SIH26066 — OceanEmbed Production Acquisition Engine
Handles robust, resumable, cached, and authenticated data fetching from
Copernicus Marine and Coriolis GDAC APIs.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List
import xarray as xr
import copernicusmarine as cm

from src.data.catalog import DATA_CATALOG, DatasetSpec, TARGET_GRID

logger = logging.getLogger("OceanEmbed.Acquisition")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class AcquisitionError(Exception):
    """Raised when data acquisition fails after all retry attempts."""
    pass


class DateNotFoundError(AcquisitionError):
    """Raised when a dataset does not contain the requested date."""
    pass


class CopernicusAcquisitionEngine:
    """
    Production-grade Copernicus Marine acquisition worker with local caching,
    exponential-backoff retries, and strict data validation.
    """
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def _get_cache_path(self, dataset_key: str, start_date: str, end_date: str) -> str:
        var_dir = os.path.join(self.raw_data_dir, dataset_key.lower())
        os.makedirs(var_dir, exist_ok=True)
        return os.path.join(var_dir, f"{dataset_key.lower()}_{start_date}_{end_date}.nc")

    def _is_cache_valid(
        self,
        filepath: str,
        required_vars: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        is_weekly: bool = False
    ) -> bool:
        """
        Validates that a local NetCDF exists, is uncorrupted, contains required variables,
        AND contains required calendar dates (exact match for daily data; bounding observations for weekly data).
        Does NOT trust filenames alone.
        """
        if not os.path.exists(filepath):
            return False
        if os.path.getsize(filepath) < 1024:  # Under 1KB is likely corrupted/empty
            return False
        try:
            with xr.open_dataset(filepath) as ds:
                # 1. Verify required variables
                for v in required_vars:
                    if v not in ds.variables and v not in ds.data_vars:
                        return False
                
                # 2. Strict internal timestamp check
                if "time" in ds.coords or "time" in ds.dims:
                    times = [str(t)[:10] for t in ds["time"].values]
                    if is_weekly:
                        # Weekly datasets require bounding observations for interpolation
                        if start_date is not None and not any(t <= start_date for t in times):
                            return False
                        if end_date is not None and not any(t >= end_date for t in times):
                            return False
                    else:
                        if start_date is not None and start_date not in times:
                            return False
                        if end_date is not None and end_date not in times:
                            return False
            return True
        except Exception:
            return False

    def find_cached_slice(self, dataset_key: str, start_date: str, end_date: str) -> Optional[str]:
        """
        Searches data/raw/{dataset_key} for any cached NetCDF file whose coverage
        fully encompasses [start_date, end_date] by inspecting internal timestamps.
        """
        var_dir = os.path.join(self.raw_data_dir, dataset_key.lower())
        if not os.path.exists(var_dir):
            return None
        
        spec = DATA_CATALOG.get(dataset_key)
        req_vars = spec.variables if spec else []
        is_weekly = (spec.temporal_frequency == "weekly") if spec else False

        for fname in os.listdir(var_dir):
            if not fname.endswith(".nc"):
                continue
            fpath = os.path.join(var_dir, fname)
            if is_weekly:
                if self._is_cache_valid(fpath, req_vars, start_date=start_date, end_date=end_date, is_weekly=True):
                    return fpath
            else:
                parts = fname[:-3].split("_")
                if len(parts) >= 3:
                    f_start = parts[-2]
                    f_end = parts[-1]
                    # Candidate filename check
                    if f_start <= start_date and f_end >= end_date:
                        # Deep verification of NetCDF coordinates
                        if self._is_cache_valid(fpath, req_vars, start_date=start_date, end_date=end_date, is_weekly=False):
                            return fpath
        return None

    def fetch_dataset_slice(
        self,
        dataset_key: str,
        start_date: str,
        end_date: str,
        force_redownload: bool = False,
        max_retries: int = 3,
        backoff_sec: float = 3.0
    ) -> str:
        """
        Retrieves a spatially-subsetted slice for the specified date range.
        Returns the absolute filepath to the validated local NetCDF file.
        """
        if dataset_key not in DATA_CATALOG:
            raise ValueError(f"Unknown dataset key '{dataset_key}'. Must be one of {list(DATA_CATALOG.keys())}")

        spec: DatasetSpec = DATA_CATALOG[dataset_key]
        is_weekly = (spec.temporal_frequency == "weekly")
        cache_file = self._get_cache_path(dataset_key, start_date, end_date)

        # 1. Check direct local cache path
        if not force_redownload and self._is_cache_valid(cache_file, spec.variables, start_date=start_date, end_date=end_date, is_weekly=is_weekly):
            logger.info(f"[{dataset_key}] Found valid cached file: {cache_file} ({os.path.getsize(cache_file):,} bytes)")
            return cache_file

        # 1b. Check if an existing wider cached file already spans the requested range
        if not force_redownload:
            wider_cache = self.find_cached_slice(dataset_key, start_date, end_date)
            if wider_cache:
                logger.info(f"[{dataset_key}] Found wider cached file covering {start_date}..{end_date}: {wider_cache} ({os.path.getsize(wider_cache):,} bytes)")
                return wider_cache

        # Bounding box with 0.2° buffer to avoid boundary edge interpolation artifacts
        min_lon = TARGET_GRID.lon_min - 0.2
        max_lon = TARGET_GRID.lon_max + 0.2
        min_lat = TARGET_GRID.lat_min - 0.2
        max_lat = TARGET_GRID.lat_max + 0.2

        if is_weekly:
            from datetime import datetime, timedelta
            dt_s = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=8)
            dt_e = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=8)
            start_dt = f"{dt_s.strftime('%Y-%m-%d')}T00:00:00"
            end_dt = f"{dt_e.strftime('%Y-%m-%d')}T23:59:59"
        else:
            start_dt = f"{start_date}T00:00:00"
            end_dt = f"{end_date}T23:59:59"

        kwargs: Dict[str, Any] = {
            "dataset_id": spec.dataset_id,
            "variables": spec.variables,
            "minimum_longitude": min_lon,
            "maximum_longitude": max_lon,
            "minimum_latitude": min_lat,
            "maximum_latitude": max_lat,
            "start_datetime": start_dt,
            "end_datetime": end_dt,
            "output_directory": os.path.dirname(cache_file),
            "output_filename": os.path.basename(cache_file),
            "overwrite": True
        }

        if spec.depth_range is not None:
            kwargs["minimum_depth"] = spec.depth_range[0]
            kwargs["maximum_depth"] = spec.depth_range[1]

        # 2. Download with exponential backoff
        for attempt in range(1, max_retries + 1):
            logger.info(f"[{dataset_key}] Downloading {start_date} to {end_date} (Attempt {attempt}/{max_retries})...")
            t0 = time.time()
            try:
                cm.subset(**kwargs)
                elapsed = time.time() - t0
                
                # 3. Validate integrity
                if self._is_cache_valid(cache_file, spec.variables, start_date=start_date, end_date=end_date):
                    logger.info(f"[{dataset_key}] Download complete in {elapsed:.2f}s ({os.path.getsize(cache_file):,} bytes)")
                    return cache_file
                else:
                    raise AcquisitionError(f"Downloaded file {cache_file} failed integrity validation or missing requested dates {start_date}..{end_date}.")

            except Exception as e:
                elapsed = time.time() - t0
                logger.warning(f"[{dataset_key}] Attempt {attempt} failed in {elapsed:.2f}s: {e}")
                if attempt < max_retries:
                    sleep_time = backoff_sec * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    raise AcquisitionError(f"[{dataset_key}] All {max_retries} attempts failed to download: {e}")

        raise AcquisitionError(f"Failed to acquire {dataset_key} for {start_date} to {end_date}")
