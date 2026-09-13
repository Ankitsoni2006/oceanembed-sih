import os
import time
import requests
import xarray as xr
import numpy as np

url_base = "https://data-argo.ifremer.fr/geo/indian_ocean/2020/09/"
headers = {"User-Agent": "Mozilla/5.0"}
target_days = ["20200914_prof.nc", "20200915_prof.nc", "20200916_prof.nc"]

os.makedirs("data/argo/argo2020", exist_ok=True)

for fname in target_days:
    local_p = os.path.join("data/argo/argo2020", fname)
    if os.path.exists(local_p) and os.path.getsize(local_p) > 1000000:
        print(f"Already exists: {fname} ({os.path.getsize(local_p)} bytes)")
    else:
        success = False
        for attempt in range(1, 4):
            try:
                print(f"Downloading {fname} (attempt {attempt})...")
                r = requests.get(url_base + fname, headers=headers, timeout=40)
                r.raise_for_status()
                with open(local_p, "wb") as f:
                    f.write(r.content)
                print(f"Successfully downloaded {fname} ({os.path.getsize(local_p)} bytes)")
                success = True
                break
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                time.sleep(2)
        if not success:
            print(f"Failed to download {fname}")

    if os.path.exists(local_p) and os.path.getsize(local_p) > 1000000:
        try:
            ds = xr.open_dataset(local_p)
            lats = ds["LATITUDE"].values
            lons = ds["LONGITUDE"].values
            nio = np.where((lats >= 5.0) & (lats <= 30.0) & (lons >= 45.0) & (lons <= 105.0))[0]
            print(f"  Verified {fname}: Total={len(lats)}, NIO={len(nio)}")
            ds.close()
        except Exception as e:
            print(f"  Failed to read {fname}: {e}")
