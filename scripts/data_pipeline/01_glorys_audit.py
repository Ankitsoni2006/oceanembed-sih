import xarray as xr
import numpy as np
import sys
import time

def audit_glorys(file_path):
    print(f"--- STARTING GLORYS AUDIT FOR: {file_path} ---")
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"FAILED TO OPEN DATASET: {e}")
        sys.exit(1)

    print("\n1. DIMENSIONS & SHAPE:")
    print(ds.dims)
    
    print("\n2. COORDINATES:")
    for coord in ds.coords:
        print(f" - {coord}: {ds[coord].values.min()} to {ds[coord].values.max()} (Size: {ds[coord].size})")
        
    print("\n3. DEPTH LEVELS (NATIVE):")
    if 'depth' in ds.coords:
        print(ds.depth.values)
    else:
        print("WARNING: 'depth' coordinate not found!")

    print("\n4. VARIABLES:")
    for var in ds.data_vars:
        print(f" - {var}: dtype={ds[var].dtype}, shape={ds[var].shape}")
        if 'units' in ds[var].attrs:
            print(f"   Units: {ds[var].attrs['units']}")

    print("\n5. MISSING VALUES / NANs (Checking first time step for 'thetao'):")
    if 'thetao' in ds.data_vars:
        thetao_sample = ds['thetao'].isel(time=0).values
        total_cells = thetao_sample.size
        nan_cells = np.isnan(thetao_sample).sum()
        print(f" - Total cells (1 timestep): {total_cells}")
        print(f" - NaN cells (Land/Bathymetry mask): {nan_cells} ({(nan_cells/total_cells)*100:.2f}%)")
        print(f" - Valid Ocean cells: {total_cells - nan_cells}")
    
    # Estimate memory
    mem_gb = ds.nbytes / (1024 ** 3)
    print(f"\n6. MEMORY/STORAGE:")
    print(f" - Dataset uncompressed memory size: {mem_gb:.2f} GB")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 01_glorys_audit.py <path_to_glorys_netcdf>")
    else:
        audit_glorys(sys.argv[1])
