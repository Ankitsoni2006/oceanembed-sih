import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import numpy as np
import torch
import xarray as xr
from src.preprocessing.grid import create_target_grid, regrid_dataset, normalize_longitude
from src.preprocessing.masks import generate_valid_mask, concatenate_variables_and_masks
from src.models.oceanembed import OceanEmbedNet
from src.training.loss import MaskedMSELoss
from src.validation.argo import ArgoColocator

print("============================================================")
print("PHASE 12: SYSTEMATIC STRESS TESTING (TRYING TO BREAK EVERYTHING)")
print("============================================================")

results = []

# TEST 1: Land Pixels & 90% NaN target
try:
    print("\n--- TEST 1: Extreme NaN & Land Masking ---")
    Y_pred = torch.randn(2, 15, 101, 241, requires_grad=True)
    Y = torch.randn(2, 15, 101, 241)
    Y[:, :, :90, :] = float('nan') # 90% land NaNs
    criterion = MaskedMSELoss()
    loss = criterion(Y_pred, Y)
    assert not torch.isnan(loss), "Loss became NaN with 90% NaNs!"
    loss.backward()
    assert Y_pred.grad is not None and not torch.isnan(Y_pred.grad).any(), "Gradient leaked NaNs!"
    print("TEST 1: PASS (Handled 90% land NaNs without numerical explosion or gradient NaN leakage)")
    results.append(("Extreme NaN Target", "PASS", "CODE"))
except Exception as e:
    print(f"TEST 1: FAIL ({e})")
    results.append(("Extreme NaN Target", "FAIL", f"CODE BUG: {e}"))

# TEST 2: Missing Surface Variable (Channel Dropout)
try:
    print("\n--- TEST 2: Missing Surface Variable (Total Cloud/Sensor Outage) ---")
    raw_vars = torch.randn(2, 7, 101, 241)
    raw_vars[:, 1, :, :] = float('nan') # Channel 1 (SSS) completely missing!
    X = concatenate_variables_and_masks(raw_vars)
    assert X[:, 1].eq(0.0).all(), "Missing channel was not safely zeroed!"
    assert X[:, 8].eq(0.0).all(), "Missing channel mask was not zeroed!"
    print("TEST 2: PASS (Missing surface input safely zeroed with 0.0 mask)")
    results.append(("Missing Input Variable", "PASS", "DATA"))
except Exception as e:
    print(f"TEST 2: FAIL ({e})")
    results.append(("Missing Input Variable", "FAIL", f"CODE BUG: {e}"))

# TEST 3: Longitude Normalization (0-360 to -180-180)
try:
    print("\n--- TEST 3: Longitude 0-360 vs -180-180 ---")
    lons = np.array([10.0, 190.0, 350.0])
    ds_test = xr.Dataset(coords={"lon": lons})
    ds_norm = normalize_longitude(ds_test, lon_var="lon")
    norm_lons = ds_norm["lon"].values
    assert (norm_lons >= -180).all() and (norm_lons <= 180).all(), "Lons not in -180..180!"
    print(f"TEST 3: PASS (0..360 successfully wrapped to {norm_lons})")
    results.append(("Longitude Normalization", "PASS", "DATA"))
except Exception as e:
    print(f"TEST 3: FAIL ({e})")
    results.append(("Longitude Normalization", "FAIL", f"CODE BUG: {e}"))

# TEST 4: Latitude Ordering (Inverted / Descending Lats)
try:
    print("\n--- TEST 4: Descending Latitude Grids ---")
    inv_lats = np.linspace(35, 0, 50)
    lons = np.linspace(40, 110, 50)
    data = np.random.randn(50, 50)
    ds_inv = xr.Dataset({"var": (("lat", "lon"), data)}, coords={"lat": inv_lats, "lon": lons})
    t_lats, t_lons = create_target_grid(5, 30, 45, 105, 0.25)
    regridded = regrid_dataset(ds_inv, t_lats, t_lons, lat_var="lat", lon_var="lon")
    assert regridded.lat.values[0] == 5.0 and regridded.lat.values[-1] == 30.0, "Latitude order incorrect!"
    print("TEST 4: PASS (regrid_dataset correctly sorts descending latitudes)")
    results.append(("Latitude Ordering", "PASS", "CODE"))
except Exception as e:
    print(f"TEST 4: FAIL ({e})")
    results.append(("Latitude Ordering", "FAIL", f"CODE BUG: {e}"))

# TEST 5: Multi-Batch Execution (B=2, B=4, B=8)
try:
    print("\n--- TEST 5: Batch Sizes B=2, B=4, B=8 on OceanEmbed ---")
    model = OceanEmbedNet(in_vars=7, num_depths=15, base_features=8, embedding_dim=16)
    depth_idx = torch.arange(15)
    for B in [1, 2, 4, 8]:
        x_batch = torch.randn(B, 14, 101, 241)
        out = model(x_batch, depth_idx)
        assert out.shape == (B, 15, 101, 241), f"Unexpected shape for B={B}: {out.shape}"
    print("TEST 5: PASS (OceanEmbed dynamically scales across batch sizes 1, 2, 4, 8)")
    results.append(("Batch Scaling", "PASS", "COMPUTE"))
except Exception as e:
    print(f"TEST 5: FAIL ({e})")
    results.append(("Batch Scaling", "FAIL", f"CODE BUG: {e}"))

# TEST 6: Sensor Outliers (-999 Fill Values)
try:
    print("\n--- TEST 6: Invalid Sensor Fill Values (-999.0) ---")
    raw = torch.tensor([[[[-999.0, 25.0], [28.0, -999.0]]]])
    mask = generate_valid_mask(raw, missing_val_threshold=-100.0)
    assert mask[0, 0, 0, 0] == 0.0 and mask[0, 0, 0, 1] == 1.0, "Mask did not filter -999!"
    print("TEST 6: PASS (Fill values correctly flagged as invalid)")
    results.append(("Fill Value Handling", "PASS", "DATA"))
except Exception as e:
    print(f"TEST 6: FAIL ({e})")
    results.append(("Fill Value Handling", "FAIL", f"CODE BUG: {e}"))

# TEST 7: ARGO Float Filter (Out of Region Floats)
try:
    print("\n--- TEST 7: ARGO Floats Outside NIO Target Box ---")
    colocator = ArgoColocator()
    ds_argo = xr.open_dataset("data/argo/20221101_prof.nc")
    profiles = colocator.extract_nio_profiles(ds_argo, lat_min=5.0, lat_max=30.0, lon_min=45.0, lon_max=105.0)
    for p in profiles:
        assert 5.0 <= p["lat"] <= 30.0 and 45.0 <= p["lon"] <= 105.0, "Float outside NIO leaked in!"
    print(f"TEST 7: PASS (Strict boundary filtering passed; {len(profiles)} valid floats inside NIO)")
    results.append(("ARGO Boundary Filter", "PASS", "DATA"))
except Exception as e:
    print(f"TEST 7: FAIL ({e})")
    results.append(("ARGO Boundary Filter", "FAIL", f"CODE BUG: {e}"))

print("\n=== SUMMARY OF BREAK-TESTS ===")
for test_name, status, cat in results:
    print(f" - {test_name:30s} : {status} ({cat})")
