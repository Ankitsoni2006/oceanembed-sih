import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

# Load the real ARGO profile we downloaded
ds = xr.open_dataset("data/argo/20221101_prof.nc")
lats = ds["LATITUDE"].values
lons = ds["LONGITUDE"].values
times = ds["JULD"].values

nio_indices = np.where((lats >= 5.0) & (lats <= 30.0) & (lons >= 45.0) & (lons <= 105.0))[0]
print(f"Testing colocation on {len(nio_indices)} real ARGO floats in the NIO region...")

TARGET_DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])

for idx in nio_indices:
    lat = lats[idx]
    lon = lons[idx]
    t = str(times[idx])[:10]
    pres = ds["PRES"].values[idx]
    temp = ds["TEMP"].values[idx]
    
    # Clean valid points
    valid = (~np.isnan(pres)) & (~np.isnan(temp)) & (pres >= 0)
    p_v = pres[valid]
    t_v = temp[valid]
    
    if len(p_v) < 10:
        continue
        
    # Colocation to 0.25 deg grid
    grid_lat = round(lat * 4) / 4
    grid_lon = round(lon * 4) / 4
    
    # 1D Vertical interpolation to the 15 SIH target depths
    # Depth in meters is approximately pressure in dbar (Saunders & Fofonoff / UNESCO)
    f_interp = interp1d(p_v, t_v, bounds_error=False, fill_value="extrapolate")
    t_interpolated = f_interp(TARGET_DEPTHS)
    
    print(f"\n--- Float #{idx} ---")
    print(f"Position: Lat {lat:.4f}N, Lon {lon:.4f}E -> Colocated Grid: Lat {grid_lat:.2f}N, Lon {grid_lon:.2f}E")
    print(f"Date: {t}")
    print(f"Native points: {len(p_v)}, Depth range: {p_v.min():.1f}m - {p_v.max():.1f}m")
    print("Interpolated to 15 SIH Depths:")
    for d, temp_val in zip(TARGET_DEPTHS, t_interpolated):
        print(f"  {d:4d}m : {temp_val:6.2f} °C")
    break # Just 1 detailed profile for audit proof
