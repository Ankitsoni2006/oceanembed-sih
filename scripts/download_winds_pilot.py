import sys, os
import time
import copernicusmarine as cm
import xarray as xr

out_dir = "data/pilot"
out_file = "winds_pilot.nc"
fpath = os.path.join(out_dir, out_file)

print("--- DOWNLOADING REAL WINDS PILOT (2020-01-01) ---")
print("Target: cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H")
t0 = time.time()
cm.subset(
    dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
    variables=["eastward_wind", "northward_wind"],
    minimum_longitude=45.0,
    maximum_longitude=105.0,
    minimum_latitude=5.0,
    maximum_latitude=30.0,
    start_datetime="2020-01-01T00:00:00",
    end_datetime="2020-01-01T23:59:59",
    output_directory=out_dir,
    output_filename=out_file
)
print(f"SUCCESS: Winds saved in {time.time()-t0:.2f}s ({os.path.getsize(fpath)} bytes)")
ds = xr.open_dataset(fpath)
print(f"Verified xarray read: Dims={dict(ds.sizes)}, Vars={list(ds.data_vars.keys())}")
print(f"Wind speeds min/max: U={float(ds.eastward_wind.min()):.2f} to {float(ds.eastward_wind.max()):.2f} m/s")
