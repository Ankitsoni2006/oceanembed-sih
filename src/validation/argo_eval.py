"""
SIH26066 — OceanEmbed Independent In-Situ ARGO Validation Engine
Colocates deep-learning model predictions with unassimilated in-situ ARGO profiling floats
from Coriolis / INCOIS GDAC to provide independent observational ground truth validation.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
import numpy as np
import xarray as xr
import torch

from src.models.oceanembed import OceanEmbedNet
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.validation.argo import ArgoColocator
from src.data.catalog import TARGET_GRID, TARGET_DEPTHS

from src.preprocessing.normalization import OceanStandardScaler

logger = logging.getLogger("OceanEmbed.ArgoVal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class ArgoValidator:
    """
    Validates model predictions directly against in-situ ARGO profiling floats.
    """
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "oceanembed",
        argo_file: str = "data/argo/20221101_prof.nc",
        device: Optional[torch.device] = None,
        scaler_path: Optional[str] = "configs/scaler_params_real.json"
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.argo_file = argo_file
        self.depth_values = list(TARGET_DEPTHS.depths)
        self.colocator = ArgoColocator(target_depths=self.depth_values)

        # Load scaler
        self.scaler_path = scaler_path
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = OceanStandardScaler.load(scaler_path)
            logger.info(f"ArgoValidator loaded OceanStandardScaler from {scaler_path}")
        else:
            self.scaler = None

        # Load model
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        logger.info(f"Loading checkpoint for ARGO validation: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        num_depths = len(self.depth_values)
        if self.model_type == "oceanembed":
            model = OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128)
        elif self.model_type == "mlp":
            model = PointwiseMLP(in_vars=7, num_depths=num_depths, hidden_dim=64)
        elif self.model_type in ["simple_cnn", "cnn"]:
            model = SimpleCNNBaseline(in_vars=7, num_depths=num_depths, hidden_dim=64)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def run_validation(self, x_sample: torch.Tensor) -> Dict[str, Any]:
        """
        Extracts NIO profiles from the real ARGO NetCDF, matches with grid cells,
        and computes observational comparison metrics.
        x_sample: [1, 14, 101, 241] surface satellite observation tensor
        """
        if self.scaler is not None:
            logger.info("Applying OceanStandardScaler normalization to ARGO input tensor...")
            x_sample = self.scaler.transform(x_sample)
        logger.info(f"Opening ARGO dataset: {self.argo_file}")
        with xr.open_dataset(self.argo_file) as ds_argo:
            profiles = self.colocator.extract_nio_profiles(
                ds_argo,
                lat_min=TARGET_GRID.lat_min,
                lat_max=TARGET_GRID.lat_max,
                lon_min=TARGET_GRID.lon_min,
                lon_max=TARGET_GRID.lon_max
            )

        logger.info(f"Extracted {len(profiles)} valid ARGO profiles in the North Indian Ocean basin.")
        if len(profiles) == 0:
            logger.warning("No ARGO profiles matched the spatial bounds.")
            return {"num_profiles": 0, "metrics": {}}

        # Run model inference on surface inputs
        depth_indices = torch.arange(len(self.depth_values), device=self.device)
        with torch.no_grad():
            pred = self.model(x_sample.to(self.device), depth_indices) # [1, 15, 101, 241]
        pred_np = pred.cpu().squeeze(0).numpy() # [15, 101, 241]

        # Colocate each ARGO profile with model prediction
        depth_errors = {int(d): [] for d in self.depth_values}
        colocated_profiles = []

        for p in profiles:
            lat = p["lat"]
            lon = p["lon"]
            # Find nearest grid cell indices
            lat_idx = int(round((lat - TARGET_GRID.lat_min) / TARGET_GRID.resolution))
            lon_idx = int(round((lon - TARGET_GRID.lon_min) / TARGET_GRID.resolution))

            if not (0 <= lat_idx < TARGET_GRID.shape[0] and 0 <= lon_idx < TARGET_GRID.shape[1]):
                continue

            model_profile = pred_np[:, lat_idx, lon_idx]
            obs_profile = p["temperatures"]

            colocated_profiles.append({
                "profile_idx": p["profile_idx"],
                "lat": lat,
                "lon": lon,
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
                "model_profile": model_profile.tolist(),
                "obs_profile": [float(t) if not np.isnan(t) else None for t in obs_profile]
            })

            for d_idx, depth_m in enumerate(self.depth_values):
                obs_t = obs_profile[d_idx]
                mod_t = model_profile[d_idx]
                if not np.isnan(obs_t) and not np.isnan(mod_t):
                    depth_errors[int(depth_m)].append(mod_t - obs_t)

        # Compute per-depth statistics against independent in-situ floats
        depth_metrics = {}
        all_errs = []
        for depth_m, errs in depth_errors.items():
            if len(errs) > 0:
                errs_arr = np.array(errs)
                rmse = float(np.sqrt(np.mean(errs_arr ** 2)))
                mae = float(np.mean(np.abs(errs_arr)))
                bias = float(np.mean(errs_arr))
                depth_metrics[f"{depth_m}m"] = {
                    "count": len(errs),
                    "rmse": rmse,
                    "mae": mae,
                    "bias": bias
                }
                all_errs.extend(errs)
            else:
                depth_metrics[f"{depth_m}m"] = {"count": 0, "rmse": None, "mae": None, "bias": None}

        overall_metrics = {
            "total_observations": len(all_errs),
            "rmse": float(np.sqrt(np.mean(np.array(all_errs) ** 2))) if all_errs else None,
            "mae": float(np.mean(np.abs(np.array(all_errs)))) if all_errs else None,
            "bias": float(np.mean(np.array(all_errs))) if all_errs else None
        }

        results = {
            "validation_type": "INDEPENDENT_IN_SITU_ARGO_OBSERVATIONS",
            "source": "Coriolis / INCOIS GDAC",
            "model_type": self.model_type,
            "checkpoint": self.checkpoint_path,
            "argo_file": self.argo_file,
            "profiles_colocated": len(colocated_profiles),
            "overall_metrics": overall_metrics,
            "depth_metrics": depth_metrics,
            "colocated_samples": colocated_profiles[:10]  # Store first 10 for inspection
        }

        self._print_argo_table(depth_metrics, overall_metrics, len(colocated_profiles))
        return results

    def _print_argo_table(self, depth_metrics: Dict[str, Any], overall: Dict[str, Any], num_prof: int):
        print("\n" + "=" * 74)
        print(f"  INDEPENDENT ARGO IN-SITU OBSERVATIONAL VALIDATION ({self.model_type.upper()})")
        print(f"  Colocated Float Profiles: {num_prof} across North Indian Ocean Basin")
        print("=" * 74)
        print(f" {'Depth':>8s} | {'Profiles':>8s} | {'RMSE (°C)':>10s} | {'MAE (°C)':>10s} | {'Bias (°C)':>10s} ")
        print("-" * 74)
        for depth_str, m in depth_metrics.items():
            cnt = str(m["count"])
            rmse = f"{m['rmse']:.4f}" if m["rmse"] is not None else "N/A"
            mae = f"{m['mae']:.4f}" if m["mae"] is not None else "N/A"
            bias = f"{m['bias']:+.4f}" if m["bias"] is not None else "N/A"
            print(f" {depth_str:>8s} | {cnt:>8s} | {rmse:>10s} | {mae:>10s} | {bias:>10s} ")
        print("-" * 74)
        o_rmse = f"{overall['rmse']:.4f}" if overall['rmse'] is not None else "N/A"
        o_mae = f"{overall['mae']:.4f}" if overall['mae'] is not None else "N/A"
        o_bias = f"{overall['bias']:+.4f}" if overall['bias'] is not None else "N/A"
        print(f" {'OVERALL':>8s} | {str(overall['total_observations']):>8s} | {o_rmse:>10s} | {o_mae:>10s} | {o_bias:>10s} ")
        print("=" * 74 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SIH26066 — ARGO In-Situ Validation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model", default="oceanembed", choices=["oceanembed", "mlp", "simple_cnn"])
    parser.add_argument("--argo-file", default="data/argo/20221101_prof.nc", help="Path to ARGO prof NetCDF")
    parser.add_argument("--dataset", default="data/processed/chunk_2020_01_pilot.pt", help="Path to sample dataset chunk")
    parser.add_argument("--output-json", default="reports/argo_validation.json", help="Path to output JSON")

    args = parser.parse_args()

    # Load input sample
    raw_pt = torch.load(args.dataset, weights_only=False)
    x_sample = raw_pt["X"][0:1]  # First sample

    validator = ArgoValidator(checkpoint_path=args.checkpoint, model_type=args.model, argo_file=args.argo_file)
    results = validator.run_validation(x_sample)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"ARGO validation report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
