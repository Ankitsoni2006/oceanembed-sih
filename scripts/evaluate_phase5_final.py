"""
SIH26066 — Phase 5 Final Evaluation & Presentation Figures Engine
Evaluates the best selected OceanEmbed Phase 5 model ONCE on:
1. September 2020 GLORYS held-out test set (30 days, 4.6M scalar cells)
2. September 2020 ARGO contemporaneous in-situ benchmark (497 points)
Compares Climatology, PointwiseMLP, SimpleCNN, Original OceanEmbed, and Improved OceanEmbed.
Generates publication-ready figures under reports/phase5/figures/.
"""

import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from src.models.oceanembed import OceanEmbedNet
from src.models.oceanembed_v3_decoder import OceanEmbedNetV3_Decoder
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.preprocessing.normalization import OceanStandardScaler
from src.data.catalog import TARGET_DEPTHS
from src.evaluation.metrics import calculate_metrics, evaluate_depth_wise

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def compute_argo_metrics(pred_array, obs_array):
    valid = (~np.isnan(pred_array)) & (~np.isnan(obs_array))
    p = pred_array[valid]
    o = obs_array[valid]
    diff = p - o
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    corr = float(np.corrcoef(p, o)[0, 1]) if len(p) > 2 and np.std(p) > 1e-8 else None
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "bias": round(bias, 4), "corr": round(corr, 4) if corr else None}

def run_final_evaluation(best_model_class, best_checkpoint_path, best_model_name):
    print("=" * 80)
    print("SIH26066 — PHASE 5: FINAL INDEPENDENT EVALUATION & BENCHMARKING")
    print("=" * 80)
    print(f"Selected Best OceanEmbed Candidate: {best_model_name}")
    print(f"Checkpoint: {best_checkpoint_path}")

    os.makedirs("reports/phase5/figures", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    # 1. Load Scaler & September 2020 Test Data
    scaler_path = "configs/scaler_params_experiment_2020.json"
    scaler = OceanStandardScaler.load(scaler_path)

    test_chunk_path = "data/processed/chunk_2020_09.pt"
    test_chunk = torch.load(test_chunk_path, weights_only=False)
    x_test_raw = test_chunk["X"]
    y_test = test_chunk["Y"]
    test_dates = test_chunk["dates"]
    x_test = scaler.transform(x_test_raw)

    depth_values = list(TARGET_DEPTHS.depths)
    num_depths = len(depth_values)
    depth_indices = torch.arange(num_depths)

    # 2. Load Models
    print("\nLoading All 5 Models for Comparative Evaluation...")
    climatology_profile = torch.tensor([28.2, 28.1, 27.9, 27.5, 26.8, 25.0, 22.5, 19.8, 17.2, 15.1, 13.0, 10.5, 8.4, 7.0, 5.2]).view(1, 15, 1, 1)

    mlp = PointwiseMLP(in_vars=7, num_depths=num_depths)
    mlp.load_state_dict(torch.load("checkpoints/pointwise_mlp_best.pt", weights_only=False)["model_state_dict"])
    mlp.eval()

    cnn = SimpleCNNBaseline(in_vars=7, num_depths=num_depths)
    cnn.load_state_dict(torch.load("checkpoints/simple_cnn_best.pt", weights_only=False)["model_state_dict"])
    cnn.eval()

    oe_orig = OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128)
    oe_orig.load_state_dict(torch.load("checkpoints/oceanembed_best.pt", weights_only=False)["model_state_dict"])
    oe_orig.eval()

    # Load Best Model
    oe_best = best_model_class(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128)
    best_state = torch.load(best_checkpoint_path, weights_only=False)
    oe_best.load_state_dict(best_state["model_state_dict"])
    oe_best.eval()

    param_counts = {
        "Climatology": 0,
        "PointwiseMLP": sum(p.numel() for p in mlp.parameters() if p.requires_grad),
        "SimpleCNN": sum(p.numel() for p in cnn.parameters() if p.requires_grad),
        "Original_OceanEmbed": sum(p.numel() for p in oe_orig.parameters() if p.requires_grad),
        best_model_name: sum(p.numel() for p in oe_best.parameters() if p.requires_grad)
    }

    # 3. Predict on GLORYS September 2020 Test Set
    print("\n[Step 1/2] Evaluating on September 2020 GLORYS Held-Out Test Set (30 days)...")
    with torch.no_grad():
        mlp_preds, cnn_preds, oe_orig_preds, oe_best_preds = [], [], [], []
        for i in range(0, len(x_test), 4):
            bx = x_test[i:i+4]
            mlp_preds.append(mlp(bx, depth_indices))
            cnn_preds.append(cnn(bx, depth_indices))
            oe_orig_preds.append(oe_orig(bx, depth_indices))
            # Handle different forward signatures
            try:
                oe_best_preds.append(oe_best(bx))
            except Exception:
                oe_best_preds.append(oe_best(bx, depth_indices))

        cat_mlp = torch.cat(mlp_preds, dim=0)
        cat_cnn = torch.cat(cnn_preds, dim=0)
        cat_oe_orig = torch.cat(oe_orig_preds, dim=0)
        cat_oe_best = torch.cat(oe_best_preds, dim=0)
        cat_clim = climatology_profile.expand(len(x_test), -1, x_test.shape[2], x_test.shape[3])

    glorys_results = {
        "Climatology": calculate_metrics(cat_clim, y_test),
        "PointwiseMLP": calculate_metrics(cat_mlp, y_test),
        "SimpleCNN": calculate_metrics(cat_cnn, y_test),
        "Original_OceanEmbed": calculate_metrics(cat_oe_orig, y_test),
        best_model_name: calculate_metrics(cat_oe_best, y_test)
    }

    glorys_depth_results = {
        "Climatology": evaluate_depth_wise(cat_clim, y_test, depth_values),
        "PointwiseMLP": evaluate_depth_wise(cat_mlp, y_test, depth_values),
        "SimpleCNN": evaluate_depth_wise(cat_cnn, y_test, depth_values),
        "Original_OceanEmbed": evaluate_depth_wise(cat_oe_orig, y_test, depth_values),
        best_model_name: evaluate_depth_wise(cat_oe_best, y_test, depth_values)
    }

    print("\n--- September 2020 GLORYS Reanalysis Test Results ---")
    for m in glorys_results:
        res = glorys_results[m]
        print(f"  {m:<24}: RMSE = {res['rmse']:.4f} C | MAE = {res['mae']:.4f} C | Bias = {res['bias']:+.4f} C | r = {res['corr']:.4f}")

    # 4. Predict on September 2020 ARGO In-Situ Observations (497 points)
    print("\n[Step 2/2] Evaluating on Contemporaneous ARGO In-Situ Profiles (497 points)...")
    matching_df = pd.read_csv("reports/argo2020/argo2020_matching.csv")

    argo_preds = {
        "Climatology": [],
        "PointwiseMLP": [],
        "SimpleCNN": [],
        "Original_OceanEmbed": [],
        best_model_name: []
    }
    argo_obs = []

    for _, row in matching_df.iterrows():
        m_date = str(row["model_date"])
        d_idx = test_dates.index(m_date)
        grid_lat = float(row["grid_latitude"])
        grid_lon = float(row["grid_longitude"])
        lat_idx = int(round((grid_lat - 5.0) / 0.25))
        lon_idx = int(round((grid_lon - 45.0) / 0.25))
        depth_m = float(row["depth"])
        obs_temp = float(row["observed_temperature"])
        z_idx = depth_values.index(depth_m)

        argo_obs.append(obs_temp)
        argo_preds["Climatology"].append(float(cat_clim[d_idx, z_idx, lat_idx, lon_idx].item()))
        argo_preds["PointwiseMLP"].append(float(cat_mlp[d_idx, z_idx, lat_idx, lon_idx].item()))
        argo_preds["SimpleCNN"].append(float(cat_cnn[d_idx, z_idx, lat_idx, lon_idx].item()))
        argo_preds["Original_OceanEmbed"].append(float(cat_oe_orig[d_idx, z_idx, lat_idx, lon_idx].item()))
        argo_preds[best_model_name].append(float(cat_oe_best[d_idx, z_idx, lat_idx, lon_idx].item()))

    obs_arr = np.array(argo_obs)
    argo_overall = {}
    argo_depth = {}

    print("\n--- September 2020 ARGO In-Situ Benchmark Results (497 points) ---")
    for m in argo_preds:
        p_arr = np.array(argo_preds[m])
        argo_overall[m] = compute_argo_metrics(p_arr, obs_arr)
        print(f"  {m:<24}: RMSE = {argo_overall[m]['rmse']:.4f} C | MAE = {argo_overall[m]['mae']:.4f} C | Bias = {argo_overall[m]['bias']:+.4f} C | r = {argo_overall[m]['corr']}")

        # Per depth
        argo_depth[m] = {}
        for d in depth_values:
            d_mask = (matching_df["depth"].values == d)
            if d_mask.sum() > 0:
                argo_depth[m][f"{int(d)}m"] = compute_argo_metrics(p_arr[d_mask], obs_arr[d_mask])

    # 5. Save Final Comparison JSON
    final_comparison = {
        "timestamp": "2026-09-13T16:45:00Z",
        "best_improved_model_name": best_model_name,
        "best_checkpoint_path": best_checkpoint_path,
        "parameter_counts": param_counts,
        "glorys_september_2020_test": {
            "overall": {m: {k: round(v, 4) if v is not None else None for k, v in glorys_results[m].items()} for m in glorys_results},
            "depth_wise": {m: {f"{int(d)}m": {k: round(v, 4) if v is not None else None for k, v in glorys_depth_results[m][d].items()} for d in depth_values} for m in glorys_depth_results}
        },
        "argo_september_2020_insitu": {
            "overall": argo_overall,
            "depth_wise": argo_depth
        }
    }

    with open("reports/phase5/final_model_comparison.json", "w") as f:
        json.dump(final_comparison, f, indent=2)
    print("\nSaved reports/phase5/final_model_comparison.json")

    # 6. Generate Figures
    print("\nGenerating Presentation-Ready Figures under reports/phase5/figures/...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Fig 1: Overall RMSE Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    models = ["Climatology", "PointwiseMLP", "SimpleCNN", "Original OceanEmbed", best_model_name]
    g_rmses = [glorys_results[m]["rmse"] for m in ["Climatology", "PointwiseMLP", "SimpleCNN", "Original_OceanEmbed", best_model_name]]
    a_rmses = [argo_overall[m]["rmse"] for m in ["Climatology", "PointwiseMLP", "SimpleCNN", "Original_OceanEmbed", best_model_name]]

    x = np.arange(len(models))
    width = 0.35
    rects1 = ax.bar(x - width/2, g_rmses, width, label="GLORYS Test (Sep 2020)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, a_rmses, width, label="ARGO In-Situ (Sep 2020)", color="#e06666")

    ax.set_ylabel("RMSE (°C)", fontsize=12)
    ax.set_title("SIH26066 — Ocean Reconstruction Benchmark Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, fontsize=10)
    ax.legend(fontsize=11)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}°", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("reports/phase5/figures/fig1_model_comparison_rmse.png")
    plt.close()

    # Fig 2: Depth-Wise RMSE Curves (ARGO In-Situ)
    fig, ax = plt.subplots(figsize=(7, 9), dpi=300)
    for m, col, ls in [
        ("Climatology", "#888888", "--"),
        ("PointwiseMLP", "#ff9900", "-."),
        ("SimpleCNN", "#27ae60", "-"),
        ("Original_OceanEmbed", "#c0392b", ":"),
        (best_model_name, "#2980b9", "-")
    ]:
        d_vals = []
        r_vals = []
        for d in depth_values:
            key = f"{int(d)}m"
            if key in argo_depth[m] and argo_depth[m][key]["rmse"] is not None:
                d_vals.append(d)
                r_vals.append(argo_depth[m][key]["rmse"])
        ax.plot(r_vals, d_vals, label=m, color=col, linestyle=ls, linewidth=2.5, marker="o", markersize=5)

    ax.set_ylim(1050, -20) # Inverted ocean depth
    ax.set_xlabel("In-Situ ARGO RMSE (°C)", fontsize=12)
    ax.set_ylabel("Depth (meters)", fontsize=12)
    ax.set_title("SIH26066 — Vertical Profile of In-Situ Error", fontsize=13, fontweight="bold")
    ax.axhspan(75, 150, color="#f39c12", alpha=0.15, label="Thermocline Region (75-150m)")
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig("reports/phase5/figures/fig2_depth_wise_rmse_argo.png")
    plt.close()

    # Fig 3: Thermocline Zoom
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    thermo_depths = [75, 100, 125, 150]
    td_labels = [f"{d}m" for d in thermo_depths]
    orig_t = [argo_depth["Original_OceanEmbed"][f"{d}m"]["rmse"] for d in thermo_depths]
    best_t = [argo_depth[best_model_name][f"{d}m"]["rmse"] for d in thermo_depths]
    cnn_t  = [argo_depth["SimpleCNN"][f"{d}m"]["rmse"] for d in thermo_depths]

    x = np.arange(len(thermo_depths))
    w = 0.25
    ax.bar(x - w, orig_t, w, label="Original OceanEmbed", color="#e74c3c")
    ax.bar(x, best_t, w, label=f"Improved ({best_model_name})", color="#3498db")
    ax.bar(x + w, cnn_t, w, label="SimpleCNN Baseline", color="#2ecc71")

    ax.set_ylabel("In-Situ RMSE (°C)", fontsize=12)
    ax.set_title("Thermocline Error Resolution (75m–150m)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(td_labels, fontsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("reports/phase5/figures/fig3_thermocline_zoom.png")
    plt.close()

    print("All figures saved successfully to reports/phase5/figures/.")

if __name__ == "__main__":
    run_final_evaluation(
        best_model_class=OceanEmbedNetV3_Decoder,
        best_checkpoint_path="checkpoints/phase5/oceanembed_v3_decoder.pt",
        best_model_name="Improved_OceanEmbed_v3"
    )
