import xarray as xr
import numpy as np
import sys
import time

def test_pipeline(file_path):
    print("--- STARTING 0.25° & DEPTH INTERPOLATION TEST ---")
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"FAILED TO OPEN DATASET: {e}")
        sys.exit(1)
        
    start_time = time.time()

    # 1. Spatial Slicing (North Indian Ocean)
    # PS Requirement: 5°N–30°N, 45°E–105°E
    print("\n1. Slicing to NIO (5-30N, 45-105E)...")
    ds_nio = ds.sel(latitude=slice(5, 30), longitude=slice(45, 105))
    print(f" - Sliced shape: {ds_nio.thetao.shape}")

    # 2. 0.25° Spatial Regridding
    print("\n2. Regridding to 0.25° x 0.25°...")
    # Create the target coordinate arrays
    target_lats = np.arange(5.0, 30.25, 0.25)
    target_lons = np.arange(45.0, 105.25, 0.25)
    
    # Perform linear interpolation (bypassing xesmf for simplicity in the test, 
    # assuming regular grids. We will monitor coastal NaN bleed).
    ds_regridded = ds_nio.interp(latitude=target_lats, longitude=target_lons, method="linear")
    print(f" - Regridded shape: {ds_regridded.thetao.shape}")
    
    # 3. Depth Standardization (15 SIH Levels)
    print("\n3. Interpolating to 15 standard SIH depths...")
    sih_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    
    ds_final = ds_regridded.interp(depth=sih_depths, method="linear")
    print(f" - Final Target shape: {ds_final.thetao.shape}")
    
    # 4. Coastal/NaN Check
    print("\n4. Checking NaN behavior after interpolation...")
    sample_slice = ds_final.thetao.isel(time=0, depth=0).values
    total_cells = sample_slice.size
    nan_cells = np.isnan(sample_slice).sum()
    print(f" - Final grid size: {target_lats.size} lat x {target_lons.size} lon = {total_cells} cells")
    print(f" - Valid ocean cells at 0m: {total_cells - nan_cells}")

    end_time = time.time()
    print(f"\nPipeline Test completed in {end_time - start_time:.2f} seconds.")
    print("OUTPUT SAVED IN MEMORY. PIPELINE IS FEASIBLE.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 02_pipeline_test.py <path_to_glorys_netcdf>")
    else:
        test_pipeline(sys.argv[1])
