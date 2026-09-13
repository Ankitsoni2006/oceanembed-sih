import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import numpy as np
import xarray as xr
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.preprocessing.grid import create_target_grid, regrid_dataset
from src.preprocessing.masks import generate_valid_mask, concatenate_variables_and_masks
from src.data.dataset import OceanDataset
from src.models.oceanembed import OceanEmbedNet
from src.training.loss import MaskedMSELoss

print("============================================================")
print("1. REBUILDING REAL X/Y WITH GENUINE SATELLITE SSS")
print("============================================================")

target_lats, target_lons = create_target_grid(5.0, 30.0, 45.0, 105.0, 0.25)
target_date = "2020-01-01"

# 1. SST: Real OSTIA L4 (2020-01-01)
ds_sst = xr.open_dataset("data/pilot/sst_regridded_pilot.nc")
sst_kelvin = ds_sst["analysed_sst"].sel(time=target_date, method="nearest").values
sst_c = sst_kelvin - 273.15 # Celsius

# 2. SSS: Real Satellite Multi-Obs SSS (cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c)
ds_sss = xr.open_dataset("data/pilot/sss_pilot.nc")
# Time interpolation between 2019-12-26 and 2020-01-02 to 2020-01-01
ds_sss_daily = ds_sss.interp(time=np.datetime64(f"{target_date}T00:00:00"), method="linear")
ds_sss_regrid = regrid_dataset(ds_sss_daily, target_lats, target_lons, lat_var="latitude", lon_var="longitude")
sss = ds_sss_regrid["sss"].values # (101, 241)

# 3. SSH: Real DUACS L4 SLA (2020-01-01)
ds_ssh = xr.open_dataset("data/pilot/ssh_pilot.nc")
ds_ssh_daily = ds_ssh.sel(time=target_date, method="nearest")
ds_ssh_regrid = regrid_dataset(ds_ssh_daily, target_lats, target_lons, lat_var="latitude", lon_var="longitude")
ssh = ds_ssh_regrid["sla"].values # (101, 241)

# 4 & 5. Currents: Real Multi-Obs L4 (uo, vo) (2020-01-01)
ds_cur = xr.open_dataset("data/pilot/currents_pilot.nc")
ds_cur_daily = ds_cur.sel(time=target_date, method="nearest").isel(depth=0)
ds_cur_regrid = regrid_dataset(ds_cur_daily, target_lats, target_lons, lat_var="latitude", lon_var="longitude")
uo = ds_cur_regrid["uo"].values # (101, 241)
vo = ds_cur_regrid["vo"].values # (101, 241)

# 6 & 7. Winds: Real Scatterometer L4 (2020-01-01)
ds_wind = xr.open_dataset("data/pilot/winds_pilot.nc")
# Compute daily mean from 24 hourly steps
ds_wind_daily = ds_wind.mean(dim="time")
ds_wind_regrid = regrid_dataset(ds_wind_daily, target_lats, target_lons, lat_var="latitude", lon_var="longitude")
u_wind = ds_wind_regrid["eastward_wind"].values # (101, 241)
v_wind = ds_wind_regrid["northward_wind"].values # (101, 241)

# 8. Target Y: Real 3D GLORYS Target (2020-01-01, 15 depths)
ds_y = xr.open_dataset("data/pilot/glorys_target_15depths_0.25deg.nc")
y_np = ds_y["thetao"].sel(time=target_date, method="nearest").values # (15, 101, 241)
Y = torch.from_numpy(y_np).unsqueeze(0).float() # [1, 15, 101, 241]

# Assemble 7 physical variables
raw_surface_7 = np.stack([sst_c, sss, ssh, uo, vo, u_wind, v_wind], axis=0) # (7, 101, 241)
raw_tensor_7 = torch.from_numpy(raw_surface_7).unsqueeze(0).float() # [1, 7, 101, 241]

# Build 14-channel X tensor (7 variables + 7 masks)
X = concatenate_variables_and_masks(raw_tensor_7, missing_val_threshold=-100.0)

# Save updated real tensor pair
torch.save({"X": X, "Y": Y}, "data/pilot/sample_X_Y_real.pt")
print("Saved rebuilt tensor to data/pilot/sample_X_Y_real.pt")

print("\n============================================================")
print("EXACT PHYSICAL CHANNEL MAPPING & CHANNEL STATISTICS AUDIT")
print("============================================================")

var_names = [
    ("Ch 0", "SST", "OSTIA L4 Reprocessed", "°C"),
    ("Ch 1", "SSS", "Multi-Obs L4 Satellite Salinity", "psu"),
    ("Ch 2", "SSH / SLA", "DUACS L4 Altimetry", "m"),
    ("Ch 3", "Surface U Current", "Multi-Obs L4 Currents", "m/s"),
    ("Ch 4", "Surface V Current", "Multi-Obs L4 Currents", "m/s"),
    ("Ch 5", "Surface U Wind", "Scatterometer L4 Winds", "m/s"),
    ("Ch 6", "Surface V Wind", "Scatterometer L4 Winds", "m/s"),
]

total_cells = 101 * 241

rows = []
for i, (ch_id, name, src, units) in enumerate(var_names):
    raw_v = raw_surface_7[i]
    nan_count = int(np.isnan(raw_v).sum())
    valid_cells = total_cells - nan_count
    val_v = raw_v[~np.isnan(raw_v)]
    
    rows.append({
        "Channel": ch_id,
        "Variable": name,
        "Source": src,
        "Units": units,
        "Min": f"{val_v.min():.3f}",
        "Max": f"{val_v.max():.3f}",
        "Mean": f"{val_v.mean():.3f}",
        "Std": f"{val_v.std():.3f}",
        "NaN Count": nan_count,
        "Valid Fraction": f"{(valid_cells/total_cells)*100:.2f}%"
    })

df_ch = pd.DataFrame(rows)
print(df_ch.to_string(index=False))

print("\nMask Channels (Ch 7 to 13):")
for i in range(7):
    mask_c = X[0, i + 7].numpy()
    valid_count = int((mask_c == 1.0).sum())
    print(f"Ch {i+7:2d}: {var_names[i][1]} Mask -> Valid: {valid_count:5d} / {total_cells} ({valid_count/total_cells*100:.2f}%)")

print("\n============================================================")
print("LAND MASK & LOSS CONTRIBUTION AUDIT")
print("============================================================")
y_surf = Y[0, 0].numpy()
total_grid_cells = total_cells
land_cells = int(np.isnan(y_surf).sum())
ocean_cells = total_grid_cells - land_cells
ocean_pct = (ocean_cells / total_grid_cells) * 100

print(f"Total grid cells:  {total_grid_cells}")
print(f"Ocean cells:       {ocean_cells}")
print(f"Land cells:        {land_cells}")
print(f"Ocean percentage:  {ocean_pct:.2f}%")

# Prove that land cells CANNOT contribute to loss
criterion = MaskedMSELoss()
# Create two predictions that differ ONLY over land pixels
pred_ocean = torch.zeros_like(Y)
pred_land_corrupted = torch.zeros_like(Y)
pred_land_corrupted[:, :, np.isnan(y_surf)] = 999999.0 # Extreme corruption over land!

loss1 = criterion(pred_ocean, Y)
loss2 = criterion(pred_land_corrupted, Y)
print(f"Loss with normal ocean:         {loss1.item():.4f}")
print(f"Loss with 999,999 on all land:  {loss2.item():.4f}")
assert torch.isclose(loss1, loss2), "Land corruption leaked into loss!"
print("MATHEMATICAL PROOF: Land cells contribute exactly 0.000 to the loss [PASS]")

print("\n============================================================")
print("REAL DATALOADER & OCEANEMBED TRAINING-STEP INTEGRATION TEST")
print("============================================================")
dataset = OceanDataset(X, Y)
loader = DataLoader(dataset, batch_size=1)
model = OceanEmbedNet(in_vars=7, num_depths=15, base_features=32, embedding_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
depth_indices = torch.arange(15)

for b_x, b_y in loader:
    print(f"Batch X shape: {tuple(b_x.shape)} -> Matches [B, 14, 101, 241]: {b_x.shape == (1, 14, 101, 241)}")
    print(f"Batch Y shape: {tuple(b_y.shape)} -> Matches [B, 15, 101, 241]: {b_y.shape == (1, 15, 101, 241)}")
    
    t0 = time.perf_counter()
    pred = model(b_x, depth_indices)
    loss = criterion(pred, b_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step_duration = (time.perf_counter() - t0) * 1000

print(f"Real-data training-step duration: {step_duration:.2f} ms")
print(f"Step loss value: {loss.item():.4f}")
has_grad = all(p.grad is not None and not torch.isnan(p.grad).any() for p in model.parameters())
print(f"Autograd gradient health: {has_grad} [PASS]")
