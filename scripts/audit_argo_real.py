import requests, re, os
import numpy as np
import xarray as xr

os.makedirs("data/argo", exist_ok=True)
url = "https://data-argo.ifremer.fr/geo/indian_ocean/2022/11/"
headers = {"User-Agent": "Mozilla/5.0"}
print("Accessing Coriolis GDAC Indian Ocean archive...")
r = requests.get(url, headers=headers, timeout=20)
files = re.findall(r'href="([^"]+\.nc)"', r.text)
print(f"Total ARGO profile files found: {len(files)}")
if files:
    target = files[0]
    local_path = os.path.join("data/argo", target)
    file_url = url + target
    print(f"Downloading sample float: {target} from {file_url}...")
    with requests.get(file_url, headers=headers, stream=True, timeout=30) as stream_resp:
        stream_resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in stream_resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded successfully: {local_path} ({os.path.getsize(local_path)} bytes)")
    
    ds = xr.open_dataset(local_path)
    print("\n=== ARGO DATASET AUDIT ===")
    print("Dimensions:", dict(ds.dims))
    print("Total Profiles in File (N_PROF):", ds.dims.get('N_PROF', 0))
    print("Total Depth/Pressure Levels (N_LEVELS):", ds.dims.get('N_LEVELS', 0))
    
    # Check spatial coverage across profiles
    lats = ds["LATITUDE"].values
    lons = ds["LONGITUDE"].values
    print(f"Latitude bounds: {np.nanmin(lats):.2f} to {np.nanmax(lats):.2f}")
    print(f"Longitude bounds: {np.nanmin(lons):.2f} to {np.nanmax(lons):.2f}")
    
    # Check NIO coverage (5-30N, 45-105E)
    nio_mask = (lats >= 5.0) & (lats <= 30.0) & (lons >= 45.0) & (lons <= 105.0)
    print(f"Profiles strictly in NIO target box (5-30N, 45-105E): {np.sum(nio_mask)} of {len(lats)}")
    
    if np.sum(nio_mask) > 0:
        idx = np.where(nio_mask)[0][0]
    else:
        idx = 0
    print(f"\nInspecting profile #{idx}:")
    print(f"Lat={lats[idx]:.4f}, Lon={lons[idx]:.4f}, Time={ds['JULD'].values[idx]}")
    
    pres = ds["PRES"].values[idx]
    temp = ds["TEMP"].values[idx]
    valid = (~np.isnan(pres)) & (~np.isnan(temp)) & (pres > 0)
    p_v = pres[valid]
    t_v = temp[valid]
    print(f"Valid vertical points: {len(p_v)}")
    print(f"Depth/Pressure Range: {p_v.min():.1f} dbar to {p_v.max():.1f} dbar")
    print(f"Temperature Range: {t_v.min():.2f} degC to {t_v.max():.2f} degC")
