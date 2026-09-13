import os
import sys
import json
import numpy as np
import xarray as xr
import torch

sys.path.insert(0, os.path.abspath("."))

# ==============================================================================
# FORENSIC ARGO AUDIT
# ==============================================================================

print("=" * 80)
print("SIH26066 — ARGO VALIDATION FORENSIC RECONCILIATION")
print("=" * 80)

argo_nc_path = "data/argo/20221101_prof.nc"
ds = xr.open_dataset(argo_nc_path)

lats = ds["LATITUDE"].values
lons = ds["LONGITUDE"].values
times = ds["JULD"].values

lat_min, lat_max = 5.0, 30.0
lon_min, lon_max = 45.0, 105.0

in_nio = np.where((lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max))[0]
print(f"\n1. RAW ARGO NETCDF INDICES IN NIO BOX: {list(in_nio)}")
print(f"   Count = {len(in_nio)}")

# Let's inspect each of these profiles in raw file:
for idx in in_nio:
    pres = ds["PRES"].values[idx]
    temp = ds["TEMP"].values[idx]
    valid = (~np.isnan(pres)) & (~np.isnan(temp)) & (pres >= 0) & (temp > -2.0) & (temp < 40.0)
    p_v = pres[valid]
    t_v = temp[valid]
    print(f"   idx={idx}: Lat={lats[idx]:.4f}, Lon={lons[idx]:.4f}, Time={str(times[idx])[:19]}, valid_pts={len(p_v)}, p_range=[{p_v.min():.1f}, {p_v.max():.1f}]")

# Check if 39, 41, 42, 45, 48 exist in ds:
print(f"\n2. CHECKING INDICES 39, 41, 42, 45, 48 in raw dataset:")
for idx in [39, 41, 42, 45, 48]:
    if idx < len(lats):
        print(f"   idx={idx}: Lat={lats[idx]:.4f}, Lon={lons[idx]:.4f}, in_nio={idx in in_nio}")
    else:
        print(f"   idx={idx}: Out of bounds (total profiles = {len(lats)})")

# Check what extract_robust_argo_profiles produces:
from scripts.validate_argo_phase4 import extract_robust_argo_profiles
target_depths = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0])
profiles = extract_robust_argo_profiles(argo_nc_path, target_depths)
print(f"\n3. PROFILES RETURNED BY extract_robust_argo_profiles():")
extracted_indices = [p["profile_idx"] for p in profiles]
print(f"   Extracted profile indices: {extracted_indices}")

# Now let's trace exactly valid counts per depth:
print(f"\n4. PER-DEPTH VALID OBSERVATION COUNT ACROSS THE 8 EXTRACTED PROFILES:")
depth_counts = {int(d): 0 for d in target_depths}
profile_depth_matrix = {}

for p in profiles:
    pid = p["profile_idx"]
    profile_depth_matrix[pid] = {}
    for d, t in zip(target_depths, p["temperatures"]):
        is_val = not np.isnan(t)
        profile_depth_matrix[pid][int(d)] = float(t) if is_val else None
        if is_val:
            depth_counts[int(d)] += 1

print(f"{'Depth':>8s} | {'Valid Count':>12s} | {'Contributing Profile Indices'}")
print("-" * 60)
for d in target_depths:
    d_int = int(d)
    contrib = [pid for pid in extracted_indices if profile_depth_matrix[pid][d_int] is not None]
    print(f"{d_int:>7d}m | {depth_counts[d_int]:>12d} | {contrib}")

sum_depth_counts = sum(depth_counts.values())
print("-" * 60)
print(f"{'SUM':>8s} | {sum_depth_counts:>12d}")

# Now let's inspect reports/results/argo_validation_results.json
print(f"\n5. INSPECTING reports/results/argo_validation_results.json:")
with open("reports/results/argo_validation_results.json") as f:
    res = json.load(f)

for mkey, mval in res["models"].items():
    ov = mval["overall"]
    print(f"   Model: {mkey}")
    print(f"     Overall: count={ov['count']}, RMSE={ov['rmse']}, MAE={ov['mae']}, Bias={ov['bias']}, Corr={ov['corr']}")
    dw = mval["depth_wise"]
    dw_sum = 0
    for dstr, dstat in dw.items():
        cnt = dstat["count"]
        dw_sum += cnt
    print(f"     Sum of depth_wise counts = {dw_sum}")
    if dw_sum != ov['count']:
        print(f"     DISCREPANCY: depth_wise sum ({dw_sum}) != overall count ({ov['count']})")
    else:
        print(f"     MATCH: depth_wise sum == overall count")

# Let's inspect validate_argo_phase4.py lines 195-238 to see how all_obs_points and depth_wise were populated:
print(f"\n6. CHECKING EXACT ACCUMULATION IN validate_argo_phase4.py:")
