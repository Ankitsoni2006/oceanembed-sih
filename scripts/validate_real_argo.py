"""
SIH26066 — Real Historical In-Situ ARGO Colocation & Validation Script
Colocates deep-learning reconstructed subsurface temperatures with real profiling floats
from Coriolis GDAC (data/argo/20221101_prof.nc).
Explicitly documents temporal provenance: 2020 surface input vs 2022 in-situ floats.
"""

import os
import sys
import json
import torch

sys.path.insert(0, os.path.abspath("."))

from src.validation.argo_eval import ArgoValidator

def run_argo_validations():
    dataset_path = "data/processed/chunk_real_2020_01_31day.pt"
    scaler_path = "configs/scaler_params_real_month.json"
    argo_file = "data/argo/20221101_prof.nc"
    reports_dir = "reports/real"
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 80)
    print("SIH26066 — IN-SITU ARGO VALIDATION AUDIT (MONTH-SCALE REAL MODELS)")
    print("=" * 80)

    raw_pt = torch.load(dataset_path, weights_only=False)
    # Use validation date 2020-01-31 (last day of month, index 30)
    sample_idx = 30
    x_sample = raw_pt["X"][sample_idx:sample_idx+1]
    eval_date = raw_pt["dates"][sample_idx]

    models = [
        ("mlp", "checkpoints/mlp_best.pt", "Pointwise MLP"),
        ("simple_cnn", "checkpoints/simple_cnn_best.pt", "Simple CNN Baseline"),
        ("oceanembed", "checkpoints/oceanembed_best.pt", "OceanEmbedNet (Proposed)")
    ]

    master_argo = {
        "dataset": dataset_path,
        "surface_observation_date": eval_date,
        "argo_file": argo_file,
        "argo_observation_date": "2022-11-01",
        "provenance_note": (
            "Cross-temporal in-situ colocation test. Demonstrates automated spatial/depth matching, "
            "quality-flag filtering, and profile extraction against Coriolis GDAC floats."
        ),
        "models": {}
    }

    for model_key, ckpt_path, display_name in models:
        print(f"\n--- Running ARGO Validation for {display_name} ---")
        validator = ArgoValidator(
            checkpoint_path=ckpt_path,
            model_type=model_key,
            argo_file=argo_file,
            scaler_path=scaler_path
        )
        res = validator.run_validation(x_sample)
        om = res.get("overall_metrics", {})
        master_argo["models"][model_key] = {
            "name": display_name,
            "overall_rmse": om.get("rmse"),
            "overall_mae": om.get("mae"),
            "overall_bias": om.get("bias"),
            "num_profiles": res.get("profiles_colocated"),
            "depth_metrics": res.get("depth_metrics")
        }

    out_path = os.path.join(reports_dir, "argo_validation.json")
    with open(out_path, "w") as f:
        json.dump(master_argo, f, indent=2)
    print(f"\nSaved master ARGO validation report to: {out_path}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Model Architecture':<28} | {'ARGO RMSE (°C)':<16} | {'ARGO MAE (°C)':<16} | {'ARGO Bias (°C)':<16}")
    print("-" * 80)
    for model_key in ["mlp", "simple_cnn", "oceanembed"]:
        m = master_argo["models"][model_key]
        print(f"{m['name']:<28} | {m['overall_rmse']:<16.4f} | {m['overall_mae']:<16.4f} | {m['overall_bias']:<16.4f}")
    print("=" * 80)

if __name__ == "__main__":
    run_argo_validations()
