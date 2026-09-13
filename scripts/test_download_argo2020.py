import os
import requests
import xarray as xr
import numpy as np

os.makedirs("data/argo/argo2020", exist_ok=True)
url_base = "https://data-argo.ifremer.fr/geo/indian_ocean/2020/09/"
headers = {"User-Agent": "Mozilla/5.0"}

# Test download of a representative day in mid-September: 2020-09-15
test_file = "20200915_prof.nc"
file_url = url_base + test_file
local_path = os.path.join("data/argo/argo2020", test_file)

print(f"Downloading {test_file} from {file_url}...")
r = requests.get(file_url, headers=headers, stream=True, timeout=30)
r.raise_for_status()

with open(local_path, "wb") as f:
    for chunk in r.iter_content(chunk_size=65536):
        if chunk:
            f.write(chunk)

size_mb = os.path.getsize(local_path) / (1024 * 1024)
print(f"Downloaded {local_path}: {size_mb:.2f} MB")

ds = xr.open_dataset(local_path)
lats = ds["LATITUDE"].values
lons = ds["LONGITUDE"].values
times = ds["JULD"].values

nio_mask = (lats >= 5.0) & (lats <= 30.0) & (lons >= 45.0) & (lons <= 105.0)
n_nio = np.sum(nio_mask)
print(f"Total profiles in file: {len(lats)}")
print(f"Profiles in North Indian Ocean (5-30N, 45-105E): {n_nio}")

if n_nio > 0:
    for idx in np.where(nio_mask)[0]:
        print(f"  Profile #{idx}: Lat={lats[idx]:.4f}, Lon={lons[idx]:.4f}, Time={times[idx]}")
ds.close()
