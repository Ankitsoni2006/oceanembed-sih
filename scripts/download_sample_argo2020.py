import os
import requests
import xarray as xr
import numpy as np

url_base = "https://data-argo.ifremer.fr/geo/indian_ocean/2020/09/"
headers = {"User-Agent": "Mozilla/5.0"}
target_days = ["20200901_prof.nc", "20200908_prof.nc", "20200915_prof.nc", "20200922_prof.nc", "20200929_prof.nc"]

for fname in target_days:
    local_p = os.path.join("data/argo/argo2020", fname)
    if not os.path.exists(local_p):
        print(f"Downloading {fname}...")
        r = requests.get(url_base + fname, headers=headers, stream=True, timeout=30)
        r.raise_for_status()
        with open(local_p, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    ds = xr.open_dataset(local_p)
    lats = ds["LATITUDE"].values
    lons = ds["LONGITUDE"].values
    times = ds["JULD"].values
    nio = np.where((lats >= 5.0) & (lats <= 30.0) & (lons >= 45.0) & (lons <= 105.0))[0]
    print(f"{fname}: Total profs={len(lats)}, NIO profs={len(nio)}")
    ds.close()
