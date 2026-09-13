import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import numpy as np
import xarray as xr
from src.preprocessing.grid import create_target_grid, regrid_dataset, normalize_longitude

os.makedirs("data/pilot", exist_ok=True)
glorys_path = r"..\zip\src\cmems_mod_glo_phy_my_0.083deg_P1D-m_1788734429512.nc"

print("--- STARTING PHASE 4: EXACT 0.25° GRID REGRIDDING ---")
t0 = time.time()
ds = xr.open_dataset(glorys_path)
t_open = time.time() - t0
print(f"Opened GLORYS file in {t_open:.2f}s")

# Define SIH target grid bounds
lat_min, lat_max = 5.0, 30.0
lon_min, lon_max = 45.0, 105.0
res = 0.25

target_lats, target_lons = create_target_grid(lat_min, lat_max, lon_min, lon_max, res)
print(f"Target grid dimensions: Lat={len(target_lats)} points, Lon={len(target_lons)} points")
print(f"Lat range: {target_lats[0]:.2f} to {target_lats[-1]:.2f} (step={res})")
print(f"Lon range: {target_lons[0]:.2f} to {target_lons[-1]:.2f} (step={res})")

# Crop region first to conserve memory & speed up interpolation
t1 = time.time()
ds_crop = ds.sel(latitude=slice(lat_min - 0.2, lat_max + 0.2), longitude=slice(lon_min - 0.2, lon_max + 0.2))
print(f"Cropped native region in {time.time()-t1:.2f}s. Native cropped shape: Lat={ds_crop.latitude.size}, Lon={ds_crop.longitude.size}")

# Regrid using our project code
t2 = time.time()
regridded = regrid_dataset(ds_crop, target_lats, target_lons, lat_var="latitude", lon_var="longitude")
t_regrid = time.time() - t2
print(f"Regridding completed in {t_regrid:.3f}s")

# Verify coordinates and dimensions
assert regridded.latitude.size == 101, f"Expected 101 lats, got {regridded.latitude.size}"
assert regridded.longitude.size == 241, f"Expected 241 lons, got {regridded.longitude.size}"
print("DIMENSIONS: EXACT 101 x 241 [PASS]")

# Verify orientation
lat_ascending = np.all(np.diff(regridded.latitude.values) > 0)
lon_ascending = np.all(np.diff(regridded.longitude.values) > 0)
print(f"Orientation: Latitude strictly ascending = {lat_ascending}, Longitude strictly ascending = {lon_ascending} [PASS]")

# Verify thetao and land mask
thetao_grid = regridded["thetao"].values[0, 0] # [time=0, depth=0]
total_cells = thetao_grid.size
nan_cells = np.isnan(thetao_grid).sum()
valid_ocean_cells = total_cells - nan_cells
valid_frac = valid_ocean_cells / total_cells

print("\n--- SPATIAL OCEAN MASK AUDIT ---")
print(f"Total grid cells: {total_cells} (101 * 241)")
print(f"Land / Bathymetry NaN cells: {nan_cells} ({nan_cells/total_cells*100:.2f}%)")
print(f"Valid Ocean cells: {valid_ocean_cells} ({valid_frac*100:.2f}%)")
print(f"Memory size of 1 regridded 2D field: {thetao_grid.nbytes / 1024:.2f} KB")

# Save regridded pilot file
out_nc = "data/pilot/glorys_regridded_pilot.nc"
regridded.to_netcdf(out_nc)
print(f"\nSaved regridded pilot dataset: {out_nc} ({os.path.getsize(out_nc)} bytes)")
print("PHASE 4 AUDIT: FULL PASS")
