import xarray as xr
import numpy as np

file_path = "data/pilot/glorys_3d_pilot.nc"
print(f"=== AUDITING REAL 3D GLORYS PILOT: {file_path} ===")

ds = xr.open_dataset(file_path)

print("\n1. DIMENSION NAMES & SIZES:")
for d, s in ds.sizes.items():
    print(f" - {d}: {s}")

print("\n2. COORDINATES:")
for c in ds.coords:
    vals = ds[c].values
    if np.issubdtype(vals.dtype, np.datetime64):
        print(f" - {c}: {vals[0]} to {vals[-1]} (Count: {len(vals)})")
    else:
        print(f" - {c}: min={np.nanmin(vals):.4f}, max={np.nanmax(vals):.4f} (Count: {len(vals)})")

print("\n3. DEPTH COORDINATE AUDIT:")
depth_vals = ds["depth"].values
print(f" - Total depth levels: {len(depth_vals)}")
print(f" - Shallowest native depth: {depth_vals[0]:.4f} m")
print(f" - Deepest native depth: {depth_vals[-1]:.4f} m")
print(f" - Does it reach 1000m? {depth_vals[-1] >= 1000.0} (Deepest is {depth_vals[-1]:.2f}m)")
print(" - All native depth levels in pilot:")
for i, d in enumerate(depth_vals):
    print(f"    Level {i:2d}: {d:9.4f} m")

print("\n4. VARIABLE AUDIT (thetao):")
thetao = ds["thetao"]
print(f" - Dtype: {thetao.dtype}")
print(f" - Shape: {thetao.shape} [time, depth, latitude, longitude]")
print(f" - Units: {thetao.attrs.get('units', 'unknown')}")
print(f" - Long name: {thetao.attrs.get('long_name', 'unknown')}")

print("\n5. VALUE & NAN AUDIT (Time step 0):")
t0_thetao = thetao.isel(time=0).values
total_cells = t0_thetao.size
nan_cells = np.isnan(t0_thetao).sum()
valid_cells = total_cells - nan_cells
valid_vals = t0_thetao[~np.isnan(t0_thetao)]

print(f" - Total cells (all depths, 1 timestep): {total_cells}")
print(f" - NaN cells (Land / Bathymetry): {nan_cells} ({nan_cells/total_cells*100:.2f}%)")
print(f" - Valid Ocean cells: {valid_cells} ({valid_cells/total_cells*100:.2f}%)")
print(f" - Minimum ocean temperature: {valid_vals.min():.2f} °C")
print(f" - Maximum ocean temperature: {valid_vals.max():.2f} °C")
print(f" - Mean ocean temperature: {valid_vals.mean():.2f} °C")

print("\n6. VERTICAL PROFILE INSPECTION AT AN ARABIAN SEA POINT (Lat ~15N, Lon ~65E):")
point_ds = ds["thetao"].sel(latitude=15.0, longitude=65.0, method="nearest").isel(time=0)
print(f"Selected point: Lat={float(point_ds.latitude):.2f}°N, Lon={float(point_ds.longitude):.2f}°E")
for d, temp in zip(point_ds.depth.values, point_ds.values):
    print(f"  Depth: {d:7.2f} m -> Temp: {temp:5.2f} °C")
