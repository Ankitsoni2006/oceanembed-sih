import sys, os
import time
import copernicusmarine as cm
import xarray as xr

out_dir = "data/pilot"
os.makedirs(out_dir, exist_ok=True)

surface_jobs = [
    {
        "name": "SSH",
        "id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        "vars": ["sla"],
        "file": "ssh_pilot.nc"
    },
    {
        "name": "CURRENTS",
        "id": "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m",
        "vars": ["uo", "vo"],
        "file": "currents_pilot.nc"
    },
    {
        "name": "WINDS",
        "id": "cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H",
        "vars": ["eastward_wind", "northward_wind"],
        "file": "winds_pilot.nc"
    }
]

print("=== DOWNLOADING SURFACE SATELLITE PILOTS (2020-01-01 to 2020-01-02) ===")

for job in surface_jobs:
    fpath = os.path.join(out_dir, job["file"])
    print(f"\nDownloading {job['name']} ({job['id']})...")
    t0 = time.time()
    try:
        cm.subset(
            dataset_id=job["id"],
            variables=job["vars"],
            minimum_longitude=45.0,
            maximum_longitude=105.0,
            minimum_latitude=5.0,
            maximum_latitude=30.0,
            start_datetime="2020-01-01T00:00:00",
            end_datetime="2020-01-02T23:59:59",
            output_directory=out_dir,
            output_filename=job["file"]
        )
        print(f"SUCCESS: {job['name']} saved in {time.time()-t0:.2f}s ({os.path.getsize(fpath)} bytes)")
        ds = xr.open_dataset(fpath)
        print(f"Verified xarray read: Dims={dict(ds.sizes)}, Vars={list(ds.data_vars.keys())}")
    except Exception as e:
        print(f"FAILED {job['name']}: {e}")
