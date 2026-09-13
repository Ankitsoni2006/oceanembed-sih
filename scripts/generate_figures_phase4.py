"""
SIH26066 — Phase 4 Stage 8: Scientific Visualization & Figure Generation
Generates high-resolution presentation plots from real model predictions
on the held-out September 2020 test period:
1. GLORYS Reference vs OceanEmbed Predicted Map + Error Map (Surface & 100m Thermocline)
2. Depth-Wise RMSE Curve (Climatology vs MLP vs SimpleCNN vs OceanEmbed)
3. Depth-Wise Pearson Correlation Curve
4. Representative Vertical Temperature Profiles (Arabian Sea, Bay of Bengal, Equatorial NIO)
5. Overall Model Performance Summary (RMSE, MAE, Pearson r)
6. Subsurface Error Distribution across Ocean Stratification Regimes
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.models.oceanembed import OceanEmbedNet
from src.preprocessing.normalization import OceanStandardScaler
from src.data.catalog import TARGET_GRID, TARGET_DEPTHS


def generate_all_figures():
    print("=" * 80)
    print("SIH26066 — PHASE 4 STAGE 8: GENERATING PRESENTATION FIGURES")
    print("=" * 80)

    fig_dir = "reports/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load results JSONs
    baseline_json = "reports/results/baseline_results.json"
    oceanembed_json = "reports/results/oceanembed_results.json"

    if not os.path.exists(baseline_json) or not os.path.exists(oceanembed_json):
        raise FileNotFoundError("Results JSONs not found! Ensure Stages 5 and 6 have completed.")

    with open(baseline_json, "r") as f:
        b_res = json.load(f)
    with open(oceanembed_json, "r") as f:
        o_res = json.load(f)

    # 2. Load Models & Checkpoints
    depth_values = list(TARGET_DEPTHS.depths)
    num_depths = len(depth_values)
    depth_indices = torch.arange(num_depths)

    scaler_path = "configs/scaler_params_experiment_2020.json"
    scaler = OceanStandardScaler.load(scaler_path)

    # Models
    mlp = PointwiseMLP(in_vars=7, num_depths=num_depths)
    mlp.load_state_dict(torch.load("checkpoints/pointwise_mlp_best.pt", weights_only=False)["model_state_dict"])
    mlp.eval()

    cnn = SimpleCNNBaseline(in_vars=7, num_depths=num_depths)
    cnn.load_state_dict(torch.load("checkpoints/simple_cnn_best.pt", weights_only=False)["model_state_dict"])
    cnn.eval()

    oceanembed = OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128)
    oceanembed.load_state_dict(torch.load("checkpoints/oceanembed_best.pt", weights_only=False)["model_state_dict"])
    oceanembed.eval()

    # Load September 2020 Test Data (chunk 09)
    test_chunk_path = "data/processed/chunk_2020_09.pt"
    test_data = torch.load(test_chunk_path, weights_only=False)
    X_test = test_data["X"]
    Y_test = test_data["Y"]
    dates_test = test_data["dates"]

    # Select representative test day: Sep 15, 2020 (Day 14)
    rep_idx = 14
    rep_date = dates_test[rep_idx]
    x_raw = X_test[rep_idx:rep_idx+1]
    y_target = Y_test[rep_idx:rep_idx+1] # [1, 15, 101, 241]

    x_norm = scaler.transform(x_raw)

    with torch.no_grad():
        mlp_pred = mlp(x_norm, depth_indices).numpy()[0]
        cnn_pred = cnn(x_norm, depth_indices).numpy()[0]
        oe_pred = oceanembed(x_norm, depth_indices).numpy()[0]

    y_true = y_target.numpy()[0]
    lats = TARGET_GRID.lats
    lons = TARGET_GRID.lons

    # Ocean mask from target (non-NaN)
    ocean_mask = ~np.isnan(y_true[0])

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # =========================================================================
    # FIGURE 1: Reference GLORYS vs OceanEmbed Prediction Map & Absolute Error
    # =========================================================================
    print("Generating Figure 1: Spatial Map & Error Comparison (Surface & 100m)...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    depth_indices_to_plot = [0, 7] # 0m (Surface) and 100m (Thermocline)
    depth_labels = ["0 m (Surface)", "100 m (Thermocline)"]

    for row, (d_idx, d_label) in enumerate(zip(depth_indices_to_plot, depth_labels)):
        t_ref = np.where(ocean_mask, y_true[d_idx], np.nan)
        t_pred = np.where(ocean_mask, oe_pred[d_idx], np.nan)
        abs_err = np.abs(t_pred - t_ref)

        vmin = np.nanpercentile(t_ref, 2)
        vmax = np.nanpercentile(t_ref, 98)

        # Col 1: Reference
        im0 = axes[row, 0].pcolormesh(lons, lats, t_ref, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, shading="auto")
        axes[row, 0].set_title(f"GLORYS Reference ({d_label}) — {rep_date}", fontsize=11, fontweight="bold")
        axes[row, 0].set_ylabel("Latitude (°N)", fontsize=10)
        fig.colorbar(im0, ax=axes[row, 0], orientation="horizontal", pad=0.08, shrink=0.7, label="Temperature (°C)")

        # Col 2: OceanEmbed Prediction
        im1 = axes[row, 1].pcolormesh(lons, lats, t_pred, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, shading="auto")
        axes[row, 1].set_title(f"OceanEmbed Reconstruction ({d_label})", fontsize=11, fontweight="bold")
        fig.colorbar(im1, ax=axes[row, 1], orientation="horizontal", pad=0.08, shrink=0.7, label="Temperature (°C)")

        # Col 3: Absolute Error Map
        err_max = min(4.0, float(np.nanpercentile(abs_err, 98)))
        im2 = axes[row, 2].pcolormesh(lons, lats, abs_err, cmap="magma", vmin=0, vmax=err_max, shading="auto")
        mean_err = np.nanmean(abs_err)
        axes[row, 2].set_title(f"Absolute Error Map (|OE - Ref|) [Mean: {mean_err:.2f}°C]", fontsize=11, fontweight="bold")
        fig.colorbar(im2, ax=axes[row, 2], orientation="horizontal", pad=0.08, shrink=0.7, label="Absolute Error (°C)")

    for ax in axes.flat:
        ax.set_xlabel("Longitude (°E)", fontsize=10)
        ax.set_aspect("equal")

    fig.suptitle(f"OceanEmbed 3D Temperature Reconstruction: North Indian Ocean Basin ({rep_date})", fontsize=14, fontweight="bold")
    f1_path = os.path.join(fig_dir, "fig1_glorys_vs_oceanembed_reconstruction.png")
    fig.savefig(f1_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {f1_path}")

    # =========================================================================
    # FIGURE 2: Depth-Wise RMSE Curve Comparison Across All Models
    # =========================================================================
    print("Generating Figure 2: Depth-Wise RMSE Curves...")
    fig, ax = plt.subplots(figsize=(8, 10))

    depths_arr = np.array(depth_values)
    clim_rmse = [b_res["models"]["climatology"]["test_metrics"]["depth_wise"][f"{int(d)}m"]["rmse"] for d in depth_values]
    mlp_rmse  = [b_res["models"]["pointwise_mlp"]["test_metrics"]["depth_wise"][f"{int(d)}m"]["rmse"] for d in depth_values]
    cnn_rmse  = [b_res["models"]["simple_cnn"]["test_metrics"]["depth_wise"][f"{int(d)}m"]["rmse"] for d in depth_values]
    oe_rmse   = [o_res["test_metrics"]["depth_wise"][f"{int(d)}m"]["rmse"] for d in depth_values]

    ax.plot(clim_rmse, depths_arr, "o--", color="#7f7f7f", linewidth=2.0, markersize=6, label="Climatology Baseline")
    ax.plot(mlp_rmse, depths_arr, "s-", color="#d62728", linewidth=2.0, markersize=6, label="Pointwise MLP (Pixel-wise)")
    ax.plot(cnn_rmse, depths_arr, "^-", color="#ff7f0e", linewidth=2.0, markersize=6, label="Simple CNN (Local Convolution)")
    ax.plot(oe_rmse, depths_arr, "D-", color="#1f77b4", linewidth=2.5, markersize=7, label="OceanEmbed (Multi-Scale U-Net)")

    ax.invert_yaxis()
    major_ticks = [0, 50, 100, 150, 200, 300, 500, 700, 1000]
    ax.set_yticks(major_ticks)
    ax.set_ylabel("Depth (meters)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Test RMSE (°C) — September 2020", fontsize=12, fontweight="bold")
    ax.set_title("Subsurface Temperature Prediction Error Across All 15 Depths (0–1000m)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=11, loc="lower left")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    f2_path = os.path.join(fig_dir, "fig2_depth_wise_rmse_curves.png")
    fig.savefig(f2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {f2_path}")

    # =========================================================================
    # FIGURE 3: Depth-Wise Pearson Correlation Curve
    # =========================================================================
    print("Generating Figure 3: Depth-Wise Pearson Correlation Curves...")
    fig, ax = plt.subplots(figsize=(8, 10))

    mlp_corr = [b_res["models"]["pointwise_mlp"]["test_metrics"]["depth_wise"][f"{int(d)}m"]["corr"] for d in depth_values]
    cnn_corr = [b_res["models"]["simple_cnn"]["test_metrics"]["depth_wise"][f"{int(d)}m"]["corr"] for d in depth_values]
    oe_corr  = [o_res["test_metrics"]["depth_wise"][f"{int(d)}m"]["corr"] for d in depth_values]

    ax.plot(mlp_corr, depths_arr, "s-", color="#d62728", linewidth=2.0, markersize=6, label="Pointwise MLP")
    ax.plot(cnn_corr, depths_arr, "^-", color="#ff7f0e", linewidth=2.0, markersize=6, label="Simple CNN")
    ax.plot(oe_corr, depths_arr, "D-", color="#1f77b4", linewidth=2.5, markersize=7, label="OceanEmbed")

    ax.invert_yaxis()
    ax.set_yticks(major_ticks)
    ax.set_ylabel("Depth (meters)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient (r)", fontsize=12, fontweight="bold")
    ax.set_title("Subsurface Temperature Correlation with GLORYS Reference", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0.4, 1.02)
    ax.axvline(0.9, color="gray", linestyle=":", label="r = 0.90 threshold")
    ax.legend(frameon=True, fontsize=11, loc="lower left")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    f3_path = os.path.join(fig_dir, "fig3_depth_wise_correlation_curves.png")
    fig.savefig(f3_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {f3_path}")

    # =========================================================================
    # FIGURE 4: Representative Vertical Profiles (4 Distinct Basins)
    # =========================================================================
    print("Generating Figure 4: Representative Vertical Temperature Profiles...")
    profile_locs = [
        ("Central Arabian Sea (16.0°N, 65.0°E)", 16.0, 65.0),
        ("Southern Bay of Bengal (10.0°N, 88.0°E)", 10.0, 88.0),
        ("Equatorial Indian Ocean (5.5°N, 78.0°E)", 5.5, 78.0),
        ("Northern Arabian Sea / Oman (22.0°N, 62.0°E)", 22.0, 62.0)
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 7), sharey=True, constrained_layout=True)

    for i, (loc_name, lat_val, lon_val) in enumerate(profile_locs):
        lat_idx = int(round((lat_val - TARGET_GRID.lat_min) / TARGET_GRID.resolution))
        lon_idx = int(round((lon_val - TARGET_GRID.lon_min) / TARGET_GRID.resolution))

        t_true_prof = y_true[:, lat_idx, lon_idx]
        t_mlp_prof  = mlp_pred[:, lat_idx, lon_idx]
        t_cnn_prof  = cnn_pred[:, lat_idx, lon_idx]
        t_oe_prof   = oe_pred[:, lat_idx, lon_idx]

        ax = axes[i]
        ax.plot(t_true_prof, depths_arr, "k-", linewidth=2.5, label="GLORYS Reference")
        ax.plot(t_mlp_prof, depths_arr, "s--", color="#d62728", linewidth=1.8, markersize=5, label="Pointwise MLP")
        ax.plot(t_cnn_prof, depths_arr, "^--", color="#ff7f0e", linewidth=1.8, markersize=5, label="Simple CNN")
        ax.plot(t_oe_prof, depths_arr, "D-", color="#1f77b4", linewidth=2.2, markersize=6, label="OceanEmbed")

        ax.set_title(loc_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperature (°C)", fontsize=11)
        if i == 0:
            ax.set_ylabel("Depth (meters)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].invert_yaxis()
    axes[0].legend(frameon=True, fontsize=10, loc="lower left")
    fig.suptitle(f"Subsurface Vertical Temperature Profile Reconstructions ({rep_date})", fontsize=14, fontweight="bold")

    f4_path = os.path.join(fig_dir, "fig4_representative_vertical_profiles.png")
    fig.savefig(f4_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {f4_path}")

    # =========================================================================
    # FIGURE 5: Overall Performance Metrics Comparison Bar Chart
    # =========================================================================
    print("Generating Figure 5: Overall Metrics Comparison Bar Chart...")
    models_list = ["Climatology", "PointwiseMLP", "SimpleCNN", "OceanEmbed"]
    rmse_vals = [
        b_res["models"]["climatology"]["test_metrics"]["overall"]["rmse"],
        b_res["models"]["pointwise_mlp"]["test_metrics"]["overall"]["rmse"],
        b_res["models"]["simple_cnn"]["test_metrics"]["overall"]["rmse"],
        o_res["test_metrics"]["overall"]["rmse"]
    ]
    mae_vals = [
        b_res["models"]["climatology"]["test_metrics"]["overall"]["mae"],
        b_res["models"]["pointwise_mlp"]["test_metrics"]["overall"]["mae"],
        b_res["models"]["simple_cnn"]["test_metrics"]["overall"]["mae"],
        o_res["test_metrics"]["overall"]["mae"]
    ]
    corr_vals = [
        b_res["models"]["climatology"]["test_metrics"]["overall"]["corr"],
        b_res["models"]["pointwise_mlp"]["test_metrics"]["overall"]["corr"],
        b_res["models"]["simple_cnn"]["test_metrics"]["overall"]["corr"],
        o_res["test_metrics"]["overall"]["corr"]
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    x = np.arange(len(models_list))
    width = 0.35

    rects1 = ax1.bar(x - width/2, rmse_vals, width, label="Test RMSE (°C)", color="#1f77b4", edgecolor="black")
    rects2 = ax1.bar(x + width/2, mae_vals, width, label="Test MAE (°C)", color="#aec7e8", edgecolor="black")
    ax1.set_ylabel("Error (°C)", fontsize=11, fontweight="bold")
    ax1.set_title("Reconstruction Error Comparison (Held-out Test Period)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_list, fontsize=11, fontweight="bold")
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Label bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Correlation bar
    rects3 = ax2.bar(models_list, corr_vals, color=["#7f7f7f", "#d62728", "#ff7f0e", "#2ca02c"], edgecolor="black", width=0.5)
    ax2.set_ylabel("Pearson Correlation (r)", fontsize=11, fontweight="bold")
    ax2.set_title("Overall Correlation with GLORYS Ground Reference", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.9, 1.0)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    f5_path = os.path.join(fig_dir, "fig5_model_performance_summary_bars.png")
    fig.savefig(f5_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {f5_path}")

    print(f"\nAll 5 presentation figures generated successfully in {fig_dir}!")


if __name__ == "__main__":
    generate_all_figures()
