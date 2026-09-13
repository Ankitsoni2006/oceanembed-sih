import os
import requests
import xarray as xr
import numpy as np

url = "https://data-argo.ifremer.fr/geo/indian_ocean/2020/09/20200925_prof.nc"
local_p = "data/argo/argo2020/20200925_prof.nc"
headers = {"User-Agent": "Mozilla/5.0"}

if not os.path.exists(local_p):
    print("Downloading 20200925_prof.nc...")
    session = requests.Session()
    r = session.get(url, headers=headers, stream=True, timeout=60)
    r.raise_for_status()
    with open(local_p, "wb") as f:
        for chunk in r.iter_content(chunk_size=131072):
            if chunk:
                f.write(chunk)
    print(f"Downloaded 20200925_prof.nc: {os.path.getsize(local_p)} bytes")

try:
    ds = xr.open_dataset(local_p)
    lats, lons = ds["LATITUDE"].values, ds["LONGITUDE"].values
    nio = np.where((lats >= 5.0) & (lats <= 30.0) & (lons >= 45.0) & (lons <= 105.0))[0]
    print(f"20200925_prof.nc: Total profs={len(lats)}, NIO profs={len(nio)}")
    ds.close()
except Exception as e:
    print(f"Error on 20200925: {e}")
