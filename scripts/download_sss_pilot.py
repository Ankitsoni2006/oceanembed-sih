import sys, os
import time
import copernicusmarine as cm
import xarray as xr
import numpy as np

out_dir = "data/pilot"
out_file = "sss_pilot.nc"
fpath = os.path.join(out_dir, out_file)

print("--- DOWNLOADING REAL SSS SATELLITE PILOT ---")
print("Target: cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c")
print("Variable: sss")
print("Bounding Box: Lat [5, 30], Lon [45, 105]")
print("Dates: 2019-12-25 to 2020-01-08")

t0 = time.time()
try:
    cm.subset(
        dataset_id="cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c",
        variables=["sss"],
        minimum_longitude=45.0,
        maximum_longitude=105.0,
        minimum_latitude=5.0,
        maximum_latitude=30.0,
        start_datetime="2019-12-25T00:00:00",
        end_datetime="2020-01-08T23:59:59",
        output_directory=out_dir,
        output_filename=out_file
    )
    print(f"SUCCESS: SSS saved in {time.time()-t0:.2f}s ({os.path.getsize(fpath)} bytes)")
    ds = xr.open_dataset(fpath)
    print(f"Verified xarray read: Dims={dict(ds.sizes)}, Vars={list(ds.data_vars.keys())}")
    print(f"Time stamps: {list(ds.time.values)}")
    print(f"SSS values: min={float(np.nanmin(ds.sss)):.2f}, max={float(np.nanmax(ds.sss)):.2f}, mean={float(np.nanmean(ds.sss)):.2f} psu")
except Exception as e:
    print(f"FAILED SSS: {type(e)} - {e}")
