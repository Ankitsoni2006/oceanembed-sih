"""
SIH26066 — Real Historical Benchmark Evaluation Script
Evaluates Pointwise MLP, Simple CNN, and OceanEmbedNet on the exact
chronological validation partition of chunk_real_2020_01_w1.pt (2020-01-06 to 2020-01-07).
Strictly separates training from testing and logs depth-wise scientific metrics.
"""

import os
import sys
import json
import torch
import numpy as np

# Ensure root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.evaluation.spatial_eval import SpatialEvaluator
from src.data.catalog import TARGET_DEPTHS

def run_benchmarks():
    dataset_path = "data/processed/chunk_real_2020_01_w1.pt"
    scaler_path = "configs/scaler_params_real.json"
    reports_dir = "reports/real"
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 80)
    print("SIH26066 — REAL HISTORICAL MODEL BENCHMARK (VALIDATION PARTITION)")
    print("=" * 80)

    raw_pt = torch.load(dataset_path, weights_only=False)
    X = raw_pt["X"]
    Y = raw_pt["Y"]
    dates = raw_pt["dates"]

    # Chronological validation partition: Days 6 and 7 (indices 5 and 6)
    val_indices = [5, 6]
    X_val = X[val_indices]
    Y_val = Y[val_indices]
    dates_val = [dates[i] for i in val_indices]

    print(f"Dataset:            {dataset_path}")
    print(f"Total chunk days:   {len(dates)} ({dates[0]} to {dates[-1]})")
    print(f"Training partition: {dates[0]} to {dates[4]} (5 days)")
    print(f"Validation partition: {dates_val[0]} to {dates_val[-1]} (2 days)")
    print(f"Scaler parameters:  {scaler_path}")
    print(f"Evaluation samples: X={list(X_val.shape)}, Y={list(Y_val.shape)}")

    models = [
        ("mlp", "checkpoints/mlp_best.pt", "Pointwise MLP"),
        ("simple_cnn", "checkpoints/simple_cnn_best.pt", "Simple CNN Baseline"),
        ("oceanembed", "checkpoints/oceanembed_best.pt", "OceanEmbedNet (Proposed)")
    ]

    all_results = {
        "dataset": dataset_path,
        "train_period": {"start": dates[0], "end": dates[4], "num_days": 5},
        "val_period": {"start": dates_val[0], "end": dates_val[-1], "num_days": 2},
        "validation_dates": dates_val,
        "scaler_used": scaler_path,
        "models": {}
    }

    depth_labels = [f"{int(d)}m" for d in TARGET_DEPTHS.depths]

    for model_key, ckpt_path, display_name in models:
        print(f"\n--- Evaluating {display_name} ({ckpt_path}) ---")
        evaluator = SpatialEvaluator(
            checkpoint_path=ckpt_path,
            model_type=model_key,
            scaler_path=scaler_path
        )
        
        summary = evaluator.evaluate_dataset(X_val, Y_val, dates_val)
        
        # Save individual model report
        ind_report_path = os.path.join(reports_dir, f"eval_{model_key}.json")
        with open(ind_report_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved individual report: {ind_report_path}")

        all_results["models"][model_key] = {
            "name": display_name,
            "checkpoint": ckpt_path,
            "overall_metrics": summary["overall_metrics"],
            "depth_wise_metrics": summary["depth_metrics"]
        }

    # Save master benchmark report
    master_path = os.path.join(reports_dir, "real_models_benchmark.json")
    with open(master_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMaster benchmark report saved to: {master_path}")

    # Print Summary Table
    print("\n" + "=" * 80)
    print(f"{'Model Architecture':<28} | {'Val RMSE (°C)':<14} | {'Val MAE (°C)':<14} | {'Val Bias (°C)':<14} | {'Pearson r':<10}")
    print("-" * 80)
    for model_key in ["mlp", "simple_cnn", "oceanembed"]:
        m_info = all_results["models"][model_key]
        m = m_info["overall_metrics"]
        print(f"{m_info['name']:<28} | {m['rmse']:<14.4f} | {m['mae']:<14.4f} | {m['bias']:<14.4f} | {m['corr']:<10.4f}")
    print("=" * 80)

    # Print Depth-Wise RMSE Table
    print("\n" + "=" * 80)
    print("DEPTH-WISE VALIDATION RMSE (°C) COMPARISON")
    print("=" * 80)
    header = f"{'Depth':<8} | " + " | ".join([f"{all_results['models'][k]['name'][:15]:<15}" for k in ['mlp', 'simple_cnn', 'oceanembed']])
    print(header)
    print("-" * len(header))
    for d_str in depth_labels:
        row = f"{d_str:<8} | "
        vals = []
        for k in ['mlp', 'simple_cnn', 'oceanembed']:
            v = all_results["models"][k]["depth_wise_metrics"].get(d_str, {}).get("rmse", float("nan"))
            vals.append(f"{v:<15.4f}")
        row += " | ".join(vals)
        print(row)
    print("=" * 80)

if __name__ == "__main__":
    run_benchmarks()
