import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

class ArgoColocator:
    """
    Adapter framework for colocating model grids with actual ARGO float profiles.
    Operates on real ARGO NetCDF files from Coriolis / INCOIS GDAC.
    """
    def __init__(self, target_depths=None, spatial_tol_deg=0.25):
        if target_depths is None:
            self.target_depths = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])
        else:
            self.target_depths = np.array(target_depths)
        self.spatial_tol = spatial_tol_deg

    def extract_nio_profiles(self, argo_ds, lat_min=5.0, lat_max=30.0, lon_min=45.0, lon_max=105.0):
        """
        Extracts all valid profiles within the target bounding box.
        """
        lats = argo_ds["LATITUDE"].values
        lons = argo_ds["LONGITUDE"].values
        times = argo_ds["JULD"].values
        
        in_region = np.where((lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max))[0]
        profiles = []
        for idx in in_region:
            pres = argo_ds["PRES"].values[idx]
            temp = argo_ds["TEMP"].values[idx]
            valid = (~np.isnan(pres)) & (~np.isnan(temp)) & (pres >= 0)
            p_v = pres[valid]
            t_v = temp[valid]
            if len(p_v) >= 10:
                # Interpolate to target depths
                f_interp = interp1d(p_v, t_v, bounds_error=False, fill_value="extrapolate")
                t_interp = f_interp(self.target_depths)
                profiles.append({
                    "profile_idx": int(idx),
                    "lat": float(lats[idx]),
                    "lon": float(lons[idx]),
                    "time": str(times[idx]),
                    "grid_lat": round(float(lats[idx]) * 4) / 4,
                    "grid_lon": round(float(lons[idx]) * 4) / 4,
                    "target_depths": self.target_depths,
                    "temperatures": t_interp
                })
        return profiles
