import xarray as xr
import numpy as np

def verify_dataset():
    glorys_path = "data/pilot/glorys_pilot.nc"
    sst_path = "data/pilot/sst_pilot.nc"
    
    print("=== VERIFYING GLORYS 3D ===")
    try:
        ds_g = xr.open_dataset(glorys_path)
        print(f"Dimensions: {ds_g.dims}")
        print(f"Variables: {list(ds_g.data_vars)}")
        
        depths = ds_g['depth'].values
        print(f"Number of depth levels: {len(depths)}")
        print(f"Deepest level: {depths[-1]}m")
        
        thetao = ds_g['thetao']
        print(f"Thetao Shape: {thetao.shape}")
        
        # Check NaNs at 1000m
        deep_slice = thetao.sel(depth=1000, method='nearest')
        valid_pixels = np.count_nonzero(~np.isnan(deep_slice.values))
        print(f"Valid pixels at ~1000m: {valid_pixels}")
        
    except Exception as e:
        print(f"GLORYS Error: {e}")

    print("\n=== VERIFYING SST ===")
    try:
        ds_s = xr.open_dataset(sst_path)
        print(f"Dimensions: {ds_s.dims}")
        print(f"SST Shape: {ds_s['analysed_sst'].shape}")
    except Exception as e:
        print(f"SST Error: {e}")

if __name__ == "__main__":
    verify_dataset()
