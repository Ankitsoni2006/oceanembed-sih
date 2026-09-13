import os
import sys
import json
import math
import glob
import pandas as pd
import numpy as np
import xarray as xr
import torch
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.abspath("."))
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.models.oceanembed import OceanEmbedNet
from src.preprocessing.normalization import OceanStandardScaler
from src.data.catalog import TARGET_GRID, TARGET_DEPTHS

# -----------------------------------------------------------------------------
# HARED PHYSICAL & GEOGRAPHIC UTILITIES
# -----------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def compute_metrics(predictions, observations):
    p = np.array(predictions, dtype=np.float64)
    o = np.array(observations, dtype=np.float64)
    valid = (~np.isnan(p)) & (~np.isnan(o))
    p_v = p[valid]
    o_v = o[valid]
    n = len(p_v)
    if n == 0:
        return {"count": 0, "rmse": None, "mae": None, "bias": None, "corr": None}
    diff = p_v - o_v
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    if n > 2 and np.std(p_v) > 1e-8 and np.std(o_v) > 1e-8:
        corr = float(np.corrcoef(p_v, o_v)[0, 1])
    else:
        corr = None
    return {
        "count": n,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "bias": round(bias, 4),
        "corr": round(corr, 4) if corr is not None and not np.isnan(corr) else None
    }

# -----------------------------------------------------------------------------
# MAIN VALIDATION PIPELINE
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("SIH26066 — PHASE E-I: ARGO 2020 CONTEMPORANEOUS VALIDATION PIPELINE")
    print("=" * 80)

    # 1. Load Test Dataset & Scaler
    test_chunk_path = "data/processed/chunk_2020_09.pt"
    print(f"\nLoading September 2020 held-out test data: {test_chunk_path}")
    test_chunk = torch.load(test_chunk_path, weights_only=False)
    x_test = test_chunk["X"]  # [30, 14, 101, 241]
    chunk_dates = test_chunk["dates"]
    print(f"Test slice contains {len(chunk_dates)} days from {chunk_dates[0]} to {chunk_dates[-1]}")

    scaler_path = "configs/scaler_params_experiment_2020.json"
    print(f"Loading training-only scaler: {scaler_path}")
    scaler = OceanStandardScaler.load(scaler_path)

    # Normalize entire test tensor
    x_test_norm = scaler.transform(x_test)  # [30, 14, 101, 241]

    # Target Depths
    target_depths = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0])
    num_depths = len(target_depths)
    depth_indices = torch.arange(num_depths)

    # 2. Load Checkpoints
    print("\nLoading model checkpoints...")
    climatology_profile = np.array([28.2, 28.1, 27.9, 27.5, 26.8, 25.0, 22.5, 19.8, 17.2, 15.1, 13.0, 10.5, 8.4, 7.0, 5.2])

    mlp = PointwiseMLP(in_vars=7, num_depths=num_depths)
    mlp.load_state_dict(torch.load("checkpoints/pointwise_mlp_best.pt", weights_only=False)["model_state_dict"])
    mlp.eval()

    cnn = SimpleCNNBaseline(in_vars=7, num_depths=num_depths)
    cnn.load_state_dict(torch.load("checkpoints/simple_cnn_best.pt", weights_only=False)["model_state_dict"])
    cnn.eval()

    oceanembed = OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128)
    oceanembed.load_state_dict(torch.load("checkpoints/oceanembed_best.pt", weights_only=False)["model_state_dict"])
    oceanembed.eval()

    # Precompute model predictions for each day in September (to allow fast indexing)
    print("Running batch inference for September 2020 test dates...")
    with torch.no_grad():
        pred_mlp_all = mlp(x_test_norm, depth_indices).cpu().numpy()          # [30, 15, 101, 241]
        pred_cnn_all = cnn(x_test_norm, depth_indices).cpu().numpy()          # [30, 15, 101, 241]
        pred_oe_all = oceanembed(x_test_norm, depth_indices).cpu().numpy()    # [30, 15, 101, 241]

    pred_clim_grid = np.tile(climatology_profile[:, None, None], (1, TARGET_GRID.shape[0], TARGET_GRID.shape[1])) # [15, 101, 241]

    # 3. Load ARGO 2020 files and extract profiles
    nc_files = sorted(glob.glob("data/argo/argo2020/*.nc"))
    print(f"\nProcessing {len(nc_files)} ARGO 2020 files: {nc_files}")

    argo_profiles = []
    for fpath in nc_files:
        ds = xr.open_dataset(fpath)
        lats = ds["LATITUDE"].values
        lons = ds["LONGITUDE"].values
        times = ds["JULD"].values
        wmo = ds["PLATFORM_NUMBER"].values if "PLATFORM_NUMBER" in ds else None
        cycles = ds["CYCLE_NUMBER"].values if "CYCLE_NUMBER" in ds else None

        nio_mask = (lats >= TARGET_GRID.lat_min) & (lats <= TARGET_GRID.lat_max) & (lons >= TARGET_GRID.lon_min) & (lons <= TARGET_GRID.lon_max)
        nio_indices = np.where(nio_mask)[0]

        for idx in nio_indices:
            lat = float(lats[idx])
            lon = float(lons[idx])
            time_dt = pd.to_datetime(str(times[idx]))
            
            if wmo is not None:
                wmo_str = "".join([c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in wmo[idx]]).strip()
            else:
                wmo_str = f"FLOAT_{idx}"
            cycle_val = int(cycles[idx]) if cycles is not None else -1

            pres = ds["PRES"].values[idx]
            temp = ds["TEMP"].values[idx]
            # Valid physical values in tropical North Indian Ocean (strictly >= 2.0 degC)
            valid = (~np.isnan(pres)) & (~np.isnan(temp)) & (pres >= 0) & (temp >= 2.0) & (temp < 40.0)
            p_v = pres[valid]
            t_v = temp[valid]

            if len(p_v) < 10 or np.max(t_v) < 15.0:
                continue

            # Sort by pressure
            sort_idx = np.argsort(p_v)
            p_sorted = p_v[sort_idx]
            t_sorted = t_v[sort_idx]
            p_min = float(p_sorted.min())
            p_max = float(p_sorted.max())

            # 1D linear interpolation to target depths (max 5m surface extrapolation)
            f_interp = interp1d(p_sorted, t_sorted, bounds_error=False, fill_value=np.nan)
            t_interp = f_interp(target_depths)

            # Enforce physical guardrail: mask out any depth beyond p_max + 10 dbar
            for d_i, d_val in enumerate(target_depths):
                if d_val > (p_max + 10.0):
                    t_interp[d_i] = np.nan
                if d_val < (p_min - 5.0): # Do not extrapolate more than 5m to surface
                    t_interp[d_i] = np.nan

            valid_count = int(np.sum(~np.isnan(t_interp)))
            if valid_count >= 5:
                argo_profiles.append({
                    "source_file": os.path.basename(fpath),
                    "file_idx": int(idx),
                    "wmo_float_id": wmo_str,
                    "cycle_number": cycle_val,
                    "latitude": lat,
                    "longitude": lon,
                    "timestamp": time_dt,
                    "p_min": p_min,
                    "p_max": p_max,
                    "temperatures": t_interp
                })
        ds.close()

    print(f"Extracted {len(argo_profiles)} quality-controlled vertical profiles inside North Indian Ocean.")

    # 4. Temporal & Spatial Collocation
    matching_records = []
    spatial_distances_km = []

    # Model point storage
    model_preds = {
        "climatology": [],
        "mlp": [],
        "simple_cnn": [],
        "oceanembed": []
    }
    observations = []
    depth_records = {int(d): {"obs": [], "clim": [], "mlp": [], "cnn": [], "oe": []} for d in target_depths}

    for p in argo_profiles:
        obs_time = p["timestamp"]
        obs_date_str = obs_time.strftime("%Y-%m-%d")

        # Temporal Match: find index in chunk_dates
        if obs_date_str in chunk_dates:
            day_idx = chunk_dates.index(obs_date_str)
            matched_model_date = obs_date_str
            # Model prediction represents daily mean (centered at 12:00 UTC)
            model_time_center = pd.to_datetime(f"{matched_model_date} 12:00:00")
            time_diff_hours = abs((obs_time - model_time_center).total_seconds()) / 3600.0
        else:
            # Fallback to closest day
            day_diffs = [abs((obs_time - pd.to_datetime(f"{cd} 12:00:00")).total_seconds()) for cd in chunk_dates]
            day_idx = int(np.argmin(day_diffs))
            matched_model_date = chunk_dates[day_idx]
            time_diff_hours = day_diffs[day_idx] / 3600.0

        # Spatial Mapping to 0.25 deg grid
        lat_idx = int(round((p["latitude"] - TARGET_GRID.lat_min) / TARGET_GRID.resolution))
        lon_idx = int(round((p["longitude"] - TARGET_GRID.lon_min) / TARGET_GRID.resolution))

        grid_lat = TARGET_GRID.lat_min + lat_idx * TARGET_GRID.resolution
        grid_lon = TARGET_GRID.lon_min + lon_idx * TARGET_GRID.resolution
        dist_km = haversine_km(p["latitude"], p["longitude"], grid_lat, grid_lon)
        spatial_distances_km.append(dist_km)

        # Extract predictions for this matched day and location
        oe_prof = pred_oe_all[day_idx, :, lat_idx, lon_idx]
        cnn_prof = pred_cnn_all[day_idx, :, lat_idx, lon_idx]
        mlp_prof = pred_mlp_all[day_idx, :, lat_idx, lon_idx]
        clim_prof = pred_clim_grid[:, lat_idx, lon_idx]
        obs_prof = p["temperatures"]

        for d_i, d_val in enumerate(target_depths):
            d_int = int(d_val)
            obs_t = obs_prof[d_i]
            if not np.isnan(obs_t):
                oe_t = float(oe_prof[d_i])
                cnn_t = float(cnn_prof[d_i])
                mlp_t = float(mlp_prof[d_i])
                clim_t = float(clim_prof[d_i])

                observations.append(float(obs_t))
                model_preds["oceanembed"].append(oe_t)
                model_preds["simple_cnn"].append(cnn_t)
                model_preds["mlp"].append(mlp_t)
                model_preds["climatology"].append(clim_t)

                depth_records[d_int]["obs"].append(float(obs_t))
                depth_records[d_int]["oe"].append(oe_t)
                depth_records[d_int]["cnn"].append(cnn_t)
                depth_records[d_int]["mlp"].append(mlp_t)
                depth_records[d_int]["clim"].append(clim_t)

                matching_records.append({
                    "argo_profile_id": f"{p['source_file']}_{p['file_idx']}",
                    "wmo_float_id": p["wmo_float_id"],
                    "argo_timestamp": str(obs_time),
                    "model_date": matched_model_date,
                    "time_difference_hours": round(time_diff_hours, 2),
                    "latitude": round(p["latitude"], 4),
                    "longitude": round(p["longitude"], 4),
                    "grid_latitude": round(grid_lat, 4),
                    "grid_longitude": round(grid_lon, 4),
                    "grid_distance_km": round(dist_km, 2),
                    "depth": d_int,
                    "observed_temperature": round(float(obs_t), 4),
                    "oceanembed_prediction": round(oe_t, 4),
                    "cnn_prediction": round(cnn_t, 4),
                    "mlp_prediction": round(mlp_t, 4),
                    "climatology_prediction": round(clim_t, 4)
                })

    # Save matching CSV
    os.makedirs("reports/argo2020", exist_ok=True)
    df_matching = pd.DataFrame(matching_records)
    csv_path = "reports/argo2020/argo2020_matching.csv"
    df_matching.to_csv(csv_path, index=False)
    print(f"\nSaved temporal-spatial matching ledger to: {csv_path}")

    # Spatial statistics
    print("\n" + "=" * 80)
    print("PHASE F: SPATIAL COLLOCATION DISTANCE STATISTICS")
    print(f"  Min distance:    {np.min(spatial_distances_km):.2f} km")
    print(f"  Mean distance:   {np.mean(spatial_distances_km):.2f} km")
    print(f"  Median distance: {np.median(spatial_distances_km):.2f} km")
    print(f"  Max distance:    {np.max(spatial_distances_km):.2f} km")
    print("=" * 80)

    # Vertical Observation Counts
    print("\n" + "=" * 80)
    print("PHASE G: VERTICAL OBSERVATION COUNTS PER TARGET DEPTH")
    print(f"{'Depth':>8s} | {'Valid ARGO Observations':>24s}")
    print("-" * 36)
    total_valid_pairs = 0
    for d_val in target_depths:
        d_int = int(d_val)
        cnt = len(depth_records[d_int]["obs"])
        print(f"{d_int:>7d}m | {cnt:>24d}")
        total_valid_pairs += cnt
    print("-" * 36)
    print(f"{'TOTAL':>8s} | {total_valid_pairs:>24d}")
    print("=" * 80)

    # Compute Overall Metrics
    overall_metrics = {}
    for mkey in ["climatology", "mlp", "simple_cnn", "oceanembed"]:
        overall_metrics[mkey] = compute_metrics(model_preds[mkey], observations)

    # Depth-wise Metrics
    depth_wise_metrics = {}
    for d_val in target_depths:
        d_int = int(d_val)
        depth_wise_metrics[f"{d_int}m"] = {
            "count": len(depth_records[d_int]["obs"]),
            "climatology": compute_metrics(depth_records[d_int]["clim"], depth_records[d_int]["obs"]),
            "mlp": compute_metrics(depth_records[d_int]["mlp"], depth_records[d_int]["obs"]),
            "simple_cnn": compute_metrics(depth_records[d_int]["cnn"], depth_records[d_int]["obs"]),
            "oceanembed": compute_metrics(depth_records[d_int]["oe"], depth_records[d_int]["obs"])
        }

    # Print Summary Table
    print("\n" + "=" * 80)
    print(f"PHASE I: CONTEMPORANEOUS ARGO 2020 BENCHMARK (TOTAL POINTS = {total_valid_pairs})")
    print("=" * 80)
    print(f"{'Model':>20s} | {'RMSE (°C)':>10s} | {'MAE (°C)':>10s} | {'Bias (°C)':>10s} | {'Pearson r':>10s}")
    print("-" * 72)
    for mkey, mname in [("climatology", "Static Climatology"), ("mlp", "Pointwise MLP"), ("simple_cnn", "Simple CNN Baseline"), ("oceanembed", "OceanEmbedNet")]:
        m = overall_metrics[mkey]
        print(f"{mname:>20s} | {m['rmse']:>10.4f} | {m['mae']:>10.4f} | {m['bias']:>+10.4f} | {m['corr']:>10.4f}")
    print("=" * 80)

    # Save to JSON
    output_report = {
        "benchmark_type": "CONTEMPORANEOUS_IN_SITU_ARGO_VALIDATION_2020",
        "evaluation_timestamp": "2026-09-13T15:18:00Z",
        "dataset_summary": {
            "source": "Coriolis / INCOIS GDAC",
            "observation_dates": sorted(list(set(df_matching["model_date"]))),
            "profile_count": len(argo_profiles),
            "unique_floats_count": len(set(p["wmo_float_id"] for p in argo_profiles)),
            "total_valid_pairs": total_valid_pairs,
            "max_temporal_offset_hours": round(float(df_matching["time_difference_hours"].max()), 2),
            "mean_spatial_distance_km": round(float(np.mean(spatial_distances_km)), 2)
        },
        "overall_metrics": overall_metrics,
        "depth_wise_metrics": depth_wise_metrics,
        "spatial_distances": {
            "min_km": round(float(np.min(spatial_distances_km)), 2),
            "mean_km": round(float(np.mean(spatial_distances_km)), 2),
            "median_km": round(float(np.median(spatial_distances_km)), 2),
            "max_km": round(float(np.max(spatial_distances_km)), 2)
        }
    }

    metrics_json_path = "reports/argo2020/argo2020_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(output_report, f, indent=2)
    print(f"\nSaved metrics JSON to: {metrics_json_path}")

if __name__ == "__main__":
    main()
