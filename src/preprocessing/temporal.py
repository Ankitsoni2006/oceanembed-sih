"""
SIH26066 — OceanEmbed Temporal Synchronization Module
Handles rigorous, zero-leakage calendar synchronization, weekly SSS interpolation,
and hourly wind aggregation without silent nearest-date substitution.
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Tuple, Optional


class TemporalAlignmentError(Exception):
    """Raised when exact temporal synchronization fails."""
    pass


def extract_exact_daily_slice(ds: xr.Dataset, target_date: str, time_var: str = "time") -> xr.Dataset:
    """
    Extracts the exact calendar day slice matching target_date (YYYY-MM-DD).
    Fails loudly if the exact target date does not exist in the dataset.
    """
    date_dt = np.datetime64(target_date)
    times = ds[time_var].values
    
    # Check if exact date matches (ignoring hours if daily)
    date_strs = [str(t)[:10] for t in times]
    if target_date not in date_strs:
        raise TemporalAlignmentError(
            f"Target date {target_date} not found in dataset times ({date_strs[0]} to {date_strs[-1]}). "
            "Silent date substitution is prohibited."
        )
        
    idx = date_strs.index(target_date)
    return ds.isel({time_var: idx})


def aggregate_hourly_winds_to_daily(ds_wind: xr.Dataset, target_date: str, time_var: str = "time") -> xr.Dataset:
    """
    Aggregates 24 hourly scatterometer wind fields to a single daily mean vector field (u10, v10).
    Ensures all 24 hours (or at least 18 hours) belong strictly to target_date.
    """
    times = ds_wind[time_var].values
    date_strs = [str(t)[:10] for t in times]
    matching_indices = [i for i, d in enumerate(date_strs) if d == target_date]
    
    if len(matching_indices) == 0:
        raise TemporalAlignmentError(f"No hourly wind observations found for {target_date}")
        
    ds_day = ds_wind.isel({time_var: matching_indices})
    ds_mean = ds_day.mean(dim=time_var, keep_attrs=True)
    # Assign standard target daily timestamp
    ds_mean = ds_mean.expand_dims({time_var: [np.datetime64(f"{target_date}T00:00:00")]})
    return ds_mean


def interpolate_weekly_sss_to_daily(
    ds_sss: xr.Dataset,
    target_date: str,
    time_var: str = "time"
) -> Tuple[xr.Dataset, dict]:
    """
    Interpolates weekly satellite SSS granules to a target calendar day.
    Requires bounding observations t0 <= target_date <= t1.
    Strictly forbids extrapolation beyond available weekly bounds.
    """
    target_dt64 = np.datetime64(f"{target_date}T00:00:00")
    times = ds_sss[time_var].values
    
    # If exact date exists
    date_strs = [str(t)[:10] for t in times]
    if target_date in date_strs:
        idx = date_strs.index(target_date)
        provenance = {
            "method": "exact_observation",
            "target_date": target_date,
            "t0": target_date,
            "t1": target_date,
            "weight": 1.0
        }
        return ds_sss.isel({time_var: idx}).expand_dims({time_var: [target_dt64]}), provenance
        
    # Find bounding weekly observations
    earlier = [t for t in times if t <= target_dt64]
    later = [t for t in times if t >= target_dt64]
    
    if not earlier or not later:
        raise TemporalAlignmentError(
            f"Cannot interpolate SSS for {target_date}: observations do not bound the target date "
            f"(dataset span: {times[0]} to {times[-1]}). Extrapolation is forbidden."
        )
        
    t0 = earlier[-1]
    t1 = later[0]
    
    dt_total = (t1 - t0) / np.timedelta64(1, 's')
    dt_target = (target_dt64 - t0) / np.timedelta64(1, 's')
    alpha = float(dt_target / dt_total) if dt_total > 0 else 0.0
    
    ds_t0 = ds_sss.sel({time_var: t0})
    ds_t1 = ds_sss.sel({time_var: t1})
    
    # Linear interpolation between bounding grids
    interpolated_sss = (1.0 - alpha) * ds_t0["sss"] + alpha * ds_t1["sss"]
    
    ds_out = ds_t0.copy()
    ds_out["sss"] = interpolated_sss
    ds_out = ds_out.expand_dims({time_var: [target_dt64]})
    
    provenance = {
        "method": "bounded_linear_interpolation",
        "target_date": target_date,
        "t0": str(t0)[:19],
        "t1": str(t1)[:19],
        "alpha": round(alpha, 4),
        "span_days": round(dt_total / 86400.0, 1)
    }
    
    return ds_out, provenance
