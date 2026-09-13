import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import numpy as np
import xarray as xr
from src.preprocessing.grid import create_target_grid, regrid_dataset

sst_path = r"C:\Users\ankit soni\Downloads\METOFFICE-GLO-SST-L4-REP-OBS-SST_1789115910698.nc"

print("--- AUDITING REAL OSTIA L4 SST DATASET ---")
t0 = time.time()
ds_sst = xr.open_dataset(sst_path)
print(f"Opened SST dataset in {time.time()-t0:.2f}s")
print(f"SST Native dims: {dict(ds_sst.dims)}")
print(f"SST Variables: {list(ds_sst.data_vars.keys())}")
print(f"SST Units: {ds_sst['analysed_sst'].attrs.get('units', 'unknown')}")

target_lats, target_lons = create_target_grid(5.0, 30.0, 45.0, 105.0, 0.25)

# Crop native 0.05 deg SST
t1 = time.time()
ds_sst_crop = ds_sst.sel(latitude=slice(4.8, 30.2), longitude=slice(44.8, 105.2))
print(f"Cropped native SST in {time.time()-t1:.2f}s. Cropped size: {dict(ds_sst_crop.dims)}")

# Regrid to 0.25 deg
t2 = time.time()
sst_regridded = regrid_dataset(ds_sst_crop, target_lats, target_lons, lat_var="latitude", lon_var="longitude")
print(f"Regridded SST in {time.time()-t2:.3f}s")

sst_field = sst_regridded["analysed_sst"].values[0] # [time=0]
print(f"Regridded SST shape: {sst_field.shape} [Expected: (101, 241)]")
nan_count = np.isnan(sst_field).sum()
total = sst_field.size
valid_count = total - nan_count
print(f"SST valid ocean cells: {valid_count} ({valid_count/total*100:.2f}%)")
print(f"SST min temp: {np.nanmin(sst_field):.2f} K ({np.nanmin(sst_field)-273.15:.2f} C)")
print(f"SST max temp: {np.nanmax(sst_field):.2f} K ({np.nanmax(sst_field)-273.15:.2f} C)")

# Save regridded SST pilot
out_nc = "data/pilot/sst_regridded_pilot.nc"
sst_regridded.to_netcdf(out_nc)
print(f"Saved regridded SST pilot: {out_nc} ({os.path.getsize(out_nc)} bytes)")
print("REAL SST SATELLITE INPUT AUDIT: FULL PASS")
