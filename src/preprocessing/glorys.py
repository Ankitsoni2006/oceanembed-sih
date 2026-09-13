"""
SIH26066 — OceanEmbed GLORYS Target Generation Module
Performs horizontal regridding to 0.25° and vertical interpolation to the 15 SIH target depths.
"""

import numpy as np
import xarray as xr
from typing import Tuple, Optional

from src.data.catalog import TARGET_GRID, TARGET_DEPTHS, TargetGridSpec, VerticalGridSpec
from src.preprocessing.grid import regrid_to_target_grid, detect_coord_names


def process_glorys_3d_slice(
    ds_glorys: xr.Dataset,
    target_grid: TargetGridSpec = TARGET_GRID,
    target_depths: VerticalGridSpec = TARGET_DEPTHS,
    var_name: str = "thetao"
) -> xr.Dataset:
    """
    Standardizes a GLORYS 3D reanalysis slice onto the 15 target depths and 0.25° grid.
    
    1. Vertical interpolation:
       - Native levels extend from 0.494m to 1062.4m.
       - Surface (0m) is extrapolated from 0.494m (mixed layer uniform assumption).
       - Levels 5m..1000m are strictly interpolated within native bounds (no deep extrapolation).
    2. Horizontal regridding:
       - Regrids from 0.083° to 0.25° (101 x 241).
    3. Retains bathymetric/land NaNs.
    
    Returns: xr.Dataset with dimensions (depth: 15, latitude: 101, longitude: 241)
    """
    if var_name not in ds_glorys:
        raise KeyError(f"Variable '{var_name}' not found in GLORYS dataset vars: {list(ds_glorys.data_vars.keys())}")
        
    depth_coord = "depth" if "depth" in ds_glorys.coords else next((c for c in ds_glorys.coords if "dep" in c.lower()), None)
    if not depth_coord:
        raise KeyError(f"Depth coordinate not found in GLORYS coords: {list(ds_glorys.coords.keys())}")
        
    target_z = np.array(target_depths.depths)
    native_z = ds_glorys[depth_coord].values
    
    # Assert native depth bounds cover required 1000m
    if native_z.max() < 1000.0:
        raise ValueError(f"GLORYS dataset does not reach 1000m depth (max={native_z.max():.1f}m)")
        
    # 1. Vertical Interpolation along depth dimension
    ds_z = ds_glorys[[var_name]].interp(
        {depth_coord: target_z},
        method="linear",
        kwargs={"fill_value": "extrapolate"}  # Only extrapolates 0.494m to 0.0m
    )
    
    # 2. Horizontal Regridding to target 0.25° grid
    ds_target = regrid_to_target_grid(ds_z, target_spec=target_grid, method="linear")
    
    # Standardize coordinate names
    if depth_coord != "depth":
        ds_target = ds_target.rename({depth_coord: "depth"})
        
    return ds_target
