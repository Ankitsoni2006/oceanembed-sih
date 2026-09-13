"""
SIH26066 — OceanEmbed Spatial Grid & Regridding Module
Provides coordinate harmonization, longitude normalization, and dynamic regridding
onto the official 0.25° North Indian Ocean target grid (101 x 241).
"""

import numpy as np
import xarray as xr
from typing import Tuple, Optional
from src.data.catalog import TARGET_GRID, TargetGridSpec


def create_target_grid(
    lat_min: float = TARGET_GRID.lat_min,
    lat_max: float = TARGET_GRID.lat_max,
    lon_min: float = TARGET_GRID.lon_min,
    lon_max: float = TARGET_GRID.lon_max,
    resolution: float = TARGET_GRID.resolution
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates the strict SIH26066 target grid coordinate arrays.
    Returns: (target_lats [101], target_lons [241])
    """
    lats = np.arange(lat_min, lat_max + resolution / 2.0, resolution)
    lons = np.arange(lon_min, lon_max + resolution / 2.0, resolution)
    return lats, lons


def detect_coord_names(ds: xr.Dataset) -> Tuple[str, str]:
    """Auto-detects latitude and longitude coordinate variable names."""
    lat_names = ['lat', 'latitude', 'LAT', 'LATITUDE']
    lon_names = ['lon', 'longitude', 'LON', 'LONGITUDE']
    
    found_lat = next((name for name in lat_names if name in ds.coords or name in ds.dims), None)
    found_lon = next((name for name in lon_names if name in lon_names and (name in ds.coords or name in ds.dims)), None)
    
    if not found_lat or not found_lon:
        raise KeyError(f"Could not identify lat/lon coordinates in dataset coords: {list(ds.coords.keys())}")
    return found_lat, found_lon


def normalize_longitude(ds: xr.Dataset, lon_var: Optional[str] = None) -> xr.Dataset:
    """
    Harmonizes longitudes to standard [-180, 180) range and sorts monotonically.
    """
    if lon_var is None:
        _, lon_var = detect_coord_names(ds)
        
    ds = ds.copy()
    if (ds.coords[lon_var] > 180.0).any():
        ds.coords[lon_var] = (ds.coords[lon_var] + 180.0) % 360.0 - 180.0
    ds = ds.sortby(lon_var)
    return ds


def regrid_to_target_grid(
    ds: xr.Dataset,
    target_spec: TargetGridSpec = TARGET_GRID,
    method: str = "linear"
) -> xr.Dataset:
    """
    Regrids an xarray Dataset onto the exact SIH26066 target grid.
    Automatically discovers coordinate names, normalizes longitudes,
    sorts coordinates, and interpolates.
    """
    lat_var, lon_var = detect_coord_names(ds)
    ds = normalize_longitude(ds, lon_var=lon_var)
    
    # Ensure ascending coordinate order for monotonic interpolation
    ds = ds.sortby(lat_var)
    ds = ds.sortby(lon_var)
    
    regridded = ds.interp(
        {lat_var: target_spec.lats, lon_var: target_spec.lons},
        method=method
    )
    
    # Rename coordinates to standard 'latitude' and 'longitude' if needed
    rename_dict = {}
    if lat_var != 'latitude':
        rename_dict[lat_var] = 'latitude'
    if lon_var != 'longitude':
        rename_dict[lon_var] = 'longitude'
        
    if rename_dict:
        regridded = regrid_regridded = regridded.rename(rename_dict)
        
    return regridded


# Backward-compatible alias
def regrid_dataset(ds, target_lats, target_lons, lat_var='lat', lon_var='lon'):
    ds = ds.sortby(lat_var)
    ds = ds.sortby(lon_var)
    return ds.interp({lat_var: target_lats, lon_var: target_lons}, method='linear')
