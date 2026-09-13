import sys, os
import time
import copernicusmarine as cm
import xarray as xr
import numpy as np

out_dir = "data/pilot"
out_file = "glorys_3d_pilot.nc"
os.makedirs(out_dir, exist_ok=True)

print("--- DOWNLOADING REAL 3D GLORYS PILOT ---")
print("Target: cmems_mod_glo_phy_my_0.083deg_P1D-m")
print("Variable: thetao")
print("Bounding Box: Lat [5, 30], Lon [45, 105]")
print("Depths: 0 to 1065m")
print("Dates: 2020-01-01 to 2020-01-03")

t0 = time.time()
try:
    res = cm.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao"],
        minimum_longitude=45.0,
        maximum_longitude=105.0,
        minimum_latitude=5.0,
        maximum_latitude=30.0,
        minimum_depth=0.0,
        maximum_depth=1065.0,
        start_datetime="2020-01-01T00:00:00",
        end_datetime="2020-01-03T23:59:59",
        output_directory=out_dir,
        output_filename=out_file,
        force_download=True
    )
    t_down = time.time() - t0
    full_path = os.path.join(out_dir, out_file)
    print(f"DOWNLOAD SUCCESS! File saved to {full_path} in {t_down:.2f}s ({os.path.getsize(full_path)} bytes)")
except Exception as e:
    print(f"DOWNLOAD FAILED: {type(e)} - {e}")
