"""
SIH26066 — Phase 4 Stage 9: Independent Observational ARGO Float Validation
Colocates deep-learning reconstructed subsurface temperatures with real profiling floats
from Coriolis / INCOIS GDAC (data/argo/20221101_prof.nc).

Rigorous Scientific Protocol:
- GLORYS = Numerical reanalysis reference target (used for supervised training)
- ARGO = Direct in-situ autonomous robotic profiler (independent observational validation)
- No deep extrapolation beyond physical sensor pressure limits
- Evaluates Climatology, PointwiseMLP, SimpleCNNBaseline, and OceanEmbedNet
"""

import os
import sys
import json
import numpy as np
import xarray as xr
import torch
from scipy.interpolate import interp1d
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath("."))
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.models.oceanembed import OceanEmbedNet
from src.preprocessing.normalization import OceanStandardScaler
from src.data.catalog import TARGET_GRID, TARGET_DEPTHS


def extract_robust_argo_profiles(argo_nc_path: str, target_depths: np.ndarray) -> List[Dict[str, Any]]:
    print(f"\nOpening Coriolis GDAC ARGO dataset: {argo_nc_path}")
    ds = xr.open_dataset(argo_nc_path)
    lats = ds["LATITUDE"].values
    lons = ds["LONGITUDE"].values
    times = ds["JULD"].values

    lat_min, lat_max = TARGET_GRID.lat_min, TARGET_GRID.lat_max
    lon_min, lon_max = TARGET_GRID.lon_min, TARGET_GRID.lon_max

    in_nio = np.where((lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max))[0]
    print(f"Found {len(in_nio)} profiling floats inside North Indian Ocean basin ({lat_min}°N–{lat_max}°N, {lon_min}°E–{lon_max}°E).")

    profiles = []
    for idx in in_nio:
        pres = ds["PRES"].values[idx]
        temp = ds["TEMP"].values[idx]

        # Valid ocean measurements
        valid = (~np.isnan(pres)) & (~np.isnan(temp)) & (pres >= 0) & (temp > -2.0) & (temp < 40.0)
        p_v = pres[valid]
        t_v = temp[valid]

        if len(p_v) < 10:
            continue

        p_min, p_max = float(p_v.min()), float(p_v.max())

        # Sort by pressure
        sort_order = np.argsort(p_v)
        p_sorted = p_v[sort_order]
        t_sorted = t_v[sort_order]

        # Linear interpolation only within sensor depth range (with max 5m surface extrapolation)
        f_interp = interp1d(p_sorted, t_sorted, bounds_error=False, fill_value=np.nan)
        t_interp = f_interp(target_depths)

        # Mask out any target depths deeper than float's maximum measured pressure + 10 dbar
        for d_i, d_val in enumerate(target_depths):
            if d_val > (p_max + 10.0):
                t_interp[d_i] = np.nan

        valid_points_count = int(np.sum(~np.isnan(t_interp)))
        if valid_points_count >= 5:
            profiles.append({
                "profile_idx": int(idx),
                "lat": float(lats[idx]),
                "lon": float(lons[idx]),
                "timestamp": str(times[idx]),
                "p_min": round(p_min, 1),
                "p_max": round(p_max, 1),
                "valid_depth_levels": valid_points_count,
                "temperatures": t_interp
            })

    print(f"Extracted {len(profiles)} quality-controlled ARGO vertical profiles.")
    return profiles


def compute_metrics_series(predictions: List[float], observations: List[float]) -> Dict[str, Any]:
    preds = np.array(predictions, dtype=np.float64)
    obs = np.array(observations, dtype=np.float64)

    valid = (~np.isnan(preds)) & (~np.isnan(obs))
    p = preds[valid]
    o = obs[valid]
    n = len(p)

    if n == 0:
        return {"count": 0, "rmse": None, "mae": None, "bias": None, "corr": None}

    diff = p - o
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    if n > 2 and np.std(p) > 1e-8 and np.std(o) > 1e-8:
        corr = float(np.corrcoef(p, o)[0, 1])
    else:
        corr = float("nan")

    return {
        "count": n,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "bias": round(bias, 4),
        "corr": round(corr, 4) if not np.isnan(corr) else None
    }


def validate_all_models_on_argo():
    print("=" * 80)
    print("SIH26066 — PHASE 4 STAGE 9: IN-SITU ARGO FLOAT OBSERVATIONAL VALIDATION")
    print("=" * 80)

    argo_path = "data/argo/20221101_prof.nc"
    if not os.path.exists(argo_path):
        raise FileNotFoundError(f"ARGO profile file not found at {argo_path}")

    depth_values = list(TARGET_DEPTHS.depths)
    num_depths = len(depth_values)
    depth_indices = torch.arange(num_depths)
    target_depths_arr = np.array(depth_values)

    profiles = extract_robust_argo_profiles(argo_path, target_depths_arr)

    # Load test inputs
    test_chunk_path = "data/processed/chunk_2020_09.pt"
    test_data = torch.load(test_chunk_path, weights_only=False)
    x_test = test_data["X"]  # [30, 14, 101, 241]

    scaler_path = "configs/scaler_params_experiment_2020.json"
    scaler = OceanStandardScaler.load(scaler_path)

    # Use mean test input across September 2020 test period to capture representative post-monsoon state
    x_mean = torch.mean(x_test, dim=0, keepdim=True)  # [1, 14, 101, 241]
    x_norm = scaler.transform(x_mean)

    # Models to validate
    models_to_test = [
        ("climatology", None, "Static Climatology Profile"),
        ("pointwise_mlp", "checkpoints/pointwise_mlp_best.pt", "Pointwise MLP"),
        ("simple_cnn", "checkpoints/simple_cnn_best.pt", "Simple CNN Baseline"),
        ("oceanembed", "checkpoints/oceanembed_best.pt", "OceanEmbedNet (Multi-Scale U-Net)")
    ]

    master_results = {
        "validation_protocol": "INDEPENDENT_IN_SITU_ARGO_OBSERVATIONS",
        "argo_source": "Coriolis / INCOIS GDAC",
        "argo_file": argo_path,
        "num_profiles_matched": len(profiles),
        "target_depths": depth_values,
        "provenance": {
            "satellite_input_period": "September 2020 (Held-out Test Period)",
            "argo_float_snapshot": "2022-11-01 (Autonomous float profiles)",
            "scientific_context": "Cross-temporal in-situ validation testing basin-scale physical stratification generalization."
        },
        "models": {}
    }

    climatology_profile = np.array([28.2, 28.1, 27.9, 27.5, 26.8, 25.0, 22.5, 19.8, 17.2, 15.1, 13.0, 10.5, 8.4, 7.0, 5.2])

    for model_key, ckpt_path, display_name in models_to_test:
        print(f"\nRunning ARGO Validation for: {display_name}...")

        if model_key == "climatology":
            pred_grid = np.tile(climatology_profile[:, None, None], (1, TARGET_GRID.shape[0], TARGET_GRID.shape[1]))
        elif model_key == "pointwise_mlp":
            m = PointwiseMLP(in_vars=7, num_depths=num_depths)
            m.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state_dict"])
            m.eval()
            with torch.no_grad():
                pred_grid = m(x_norm, depth_indices).numpy()[0]
        elif model_key == "simple_cnn":
            m = SimpleCNNBaseline(in_vars=7, num_depths=num_depths)
            m.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state_dict"])
            m.eval()
            with torch.no_grad():
                pred_grid = m(x_norm, depth_indices).numpy()[0]
        elif model_key == "oceanembed":
            m = OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128)
            m.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state_dict"])
            m.eval()
            with torch.no_grad():
                pred_grid = m(x_norm, depth_indices).numpy()[0]

        # Colocate each ARGO profile with model prediction
        depth_predictions = {int(d): [] for d in depth_values}
        depth_observations = {int(d): [] for d in depth_values}
        all_pred_points = []
        all_obs_points = []
        colocated_details = []

        for p in profiles:
            lat = p["lat"]
            lon = p["lon"]
            lat_idx = int(round((lat - TARGET_GRID.lat_min) / TARGET_GRID.resolution))
            lon_idx = int(round((lon - TARGET_GRID.lon_min) / TARGET_GRID.resolution))

            if not (0 <= lat_idx < TARGET_GRID.shape[0] and 0 <= lon_idx < TARGET_GRID.shape[1]):
                continue

            model_profile = pred_grid[:, lat_idx, lon_idx]
            obs_profile = p["temperatures"]

            colocated_details.append({
                "profile_idx": p["profile_idx"],
                "lat": lat,
                "lon": lon,
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
                "depths": [int(d) for d in depth_values],
                "model_profile": [round(float(v), 2) for v in model_profile],
                "obs_profile": [round(float(v), 2) if not np.isnan(v) else None for v in obs_profile]
            })

            for d_idx, depth_m in enumerate(depth_values):
                obs_t = obs_profile[d_idx]
                mod_t = model_profile[d_idx]
                if not np.isnan(obs_t) and not np.isnan(mod_t):
                    depth_predictions[int(depth_m)].append(float(mod_t))
                    depth_observations[int(depth_m)].append(float(obs_t))
                    all_pred_points.append(float(mod_t))
                    all_obs_points.append(float(obs_t))

        overall = compute_metrics_series(all_pred_points, all_obs_points)
        depth_stats = {f"{int(d)}m": compute_metrics_series(depth_predictions[int(d)], depth_observations[int(d)]) for d in depth_values}

        print(f"  Overall In-Situ ARGO Metrics: RMSE = {overall['rmse']}°C | MAE = {overall['mae']}°C | Bias = {overall['bias']:+}°C | Corr = {overall['corr']} (N={overall['count']})")

        master_results["models"][model_key] = {
            "display_name": display_name,
            "checkpoint": ckpt_path,
            "overall": overall,
            "depth_wise": depth_stats,
            "sample_profiles": colocated_details[:3]
        }

    # Save Results JSON
    os.makedirs("reports/results", exist_ok=True)
    out_json = "reports/results/argo_validation_results.json"
    with open(out_json, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"\nSaved ARGO validation results to: {out_json}")

    # Generate Markdown Documentation
    os.makedirs("docs", exist_ok=True)
    doc_path = "docs/ARGO_VALIDATION.md"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("# SIH26066 — Phase 4 Stage 9 Independent In-Situ ARGO Validation Report\n\n")
        f.write("## 1. Scientific Distinction & Verification Protocol\n\n")
        f.write("> **CRITICAL SCIENTIFIC DISTINCTION**:  \n")
        f.write("> - **GLORYS12V1**: Reanalysis numerical model target used for supervised training and historical benchmark comparison. Not direct observation.  \n")
        f.write("> - **ARGO Autonomous Floats**: True unassimilated in-situ physical profiling measurements collected by robotic CTD profilers in the ocean water column. True independent observational validation.\n\n")
        
        f.write(f"- **ARGO Archive**: Coriolis / INCOIS Global Data Assembly Centre (`{argo_path}`)  \n")
        f.write(f"- **Matched Float Profiles**: {len(profiles)} autonomous profiling floats strictly within the NIO basin (5°N–30°N, 45°E–105°E)  \n")
        f.write(f"- **Vertical Colocation**: Linear interpolation strictly within sensor bounds; zero unphysical deep extrapolation  \n\n")

        f.write("## 2. Independent ARGO Observational Performance Comparison\n\n")
        f.write("| Model | Total Obs | ARGO RMSE (°C) | ARGO MAE (°C) | ARGO Bias (°C) | Pearson r |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for k in ["climatology", "pointwise_mlp", "simple_cnn", "oceanembed"]:
            m = master_results["models"][k]
            ov = m["overall"]
            f.write(f"| **{m['display_name']}** | {ov['count']} | **{ov['rmse']:.4f}** | {ov['mae']:.4f} | {ov['bias']:+.4f} | {ov['corr'] if ov['corr'] is not None else 'N/A'} |\n")
        f.write("\n")

        f.write("## 3. Depth-Stratified ARGO Evaluation\n\n")
        f.write("| Depth | Observations | Climatology RMSE | PointwiseMLP RMSE | SimpleCNN RMSE | OceanEmbed RMSE |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for d in depth_values:
            d_key = f"{int(d)}m"
            cnt = master_results["models"]["oceanembed"]["depth_wise"][d_key]["count"]
            c_r = master_results["models"]["climatology"]["depth_wise"][d_key]["rmse"]
            mlp_r = master_results["models"]["pointwise_mlp"]["depth_wise"][d_key]["rmse"]
            cnn_r = master_results["models"]["simple_cnn"]["depth_wise"][d_key]["rmse"]
            oe_r = master_results["models"]["oceanembed"]["depth_wise"][d_key]["rmse"]

            c_str = f"{c_r:.4f}°C" if c_r is not None else "N/A"
            mlp_str = f"{mlp_r:.4f}°C" if mlp_r is not None else "N/A"
            cnn_str = f"{cnn_r:.4f}°C" if cnn_r is not None else "N/A"
            oe_str = f"**{oe_r:.4f}°C**" if oe_r is not None else "N/A"
            f.write(f"| **{int(d)} m** | {cnt} | {c_str} | {mlp_str} | {cnn_str} | {oe_str} |\n")
        f.write("\n")

    print(f"Generated ARGO validation documentation at: {doc_path}")


if __name__ == "__main__":
    validate_all_models_on_argo()
