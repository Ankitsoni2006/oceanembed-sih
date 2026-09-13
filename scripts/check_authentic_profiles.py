import os
import glob
import xarray as xr
import numpy as np

nc_files = sorted(glob.glob("data/argo/argo2020/*.nc"))

for fpath in nc_files:
    fname = os.path.basename(fpath)
    ds = xr.open_dataset(fpath)
    lats = ds["LATITUDE"].values
    lons = ds["LONGITUDE"].values
    nio = np.where((lats >= 5.0) & (lats <= 30.0) & (lons >= 45.0) & (lons <= 105.0))[0]
    
    valid_profiles = []
    dummy_profiles = []
    
    for idx in nio:
        temp = ds["TEMP"].values[idx]
        pres = ds["PRES"].values[idx]
        val_mask = (~np.isnan(pres)) & (~np.isnan(temp)) & (pres >= 0)
        t_clean = temp[val_mask]
        
        # Check if dummy zero profile
        if len(t_clean) > 0 and np.max(t_clean) < 2.0:
            dummy_profiles.append(idx)
        elif len(t_clean) >= 10 and np.max(t_clean) > 20.0 and np.min(t_clean) >= 2.0:
            valid_profiles.append(idx)
            
    print(f"{fname}: NIO total={len(nio)}, Valid authentic={len(valid_profiles)}, Dummy zeroed={len(dummy_profiles)}")
    ds.close()
