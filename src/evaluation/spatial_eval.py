"""
SIH26066 — OceanEmbed Comprehensive Spatial & Depth Evaluation Engine
Computes depth-stratified metrics (RMSE, MAE, Bias, Pearson r) and generates
2D spatial error distributions across the North Indian Ocean basin.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.models.oceanembed import OceanEmbedNet
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.evaluation.metrics import calculate_metrics, evaluate_depth_wise
from src.data.catalog import TARGET_DEPTHS, TARGET_GRID

from src.preprocessing.normalization import OceanStandardScaler

logger = logging.getLogger("OceanEmbed.Eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class SpatialEvaluator:
    """
    Evaluator computing basin-wide and depth-stratified scientific metrics.
    """
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "oceanembed",
        device: Optional[torch.device] = None,
        depth_values: Optional[List[float]] = None,
        scaler_path: Optional[str] = "configs/scaler_params_real.json"
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.depth_values = depth_values or list(TARGET_DEPTHS.depths)
        self.num_depths = len(self.depth_values)
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path

        # Load scaler
        self.scaler_path = scaler_path
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = OceanStandardScaler.load(scaler_path)
            logger.info(f"SpatialEvaluator loaded OceanStandardScaler from {scaler_path}")
        else:
            self.scaler = None

        # Load model
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        if self.model_type == "oceanembed":
            model = OceanEmbedNet(in_vars=7, num_depths=self.num_depths, base_features=32, embedding_dim=128)
        elif self.model_type == "mlp":
            model = PointwiseMLP(in_vars=7, num_depths=self.num_depths, hidden_dim=64)
        elif self.model_type in ["simple_cnn", "cnn"]:
            model = SimpleCNNBaseline(in_vars=7, num_depths=self.num_depths, hidden_dim=64)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def evaluate_dataset(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        dates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates predictions against reference target across all samples.
        x_tensor: [N, 14, 101, 241]
        y_tensor: [N, 15, 101, 241]
        """
        if self.scaler is not None:
            logger.info("Applying OceanStandardScaler normalization to input tensors...")
            x_tensor = self.scaler.transform(x_tensor)
        N = x_tensor.shape[0]
        depth_indices = torch.arange(self.num_depths, device=self.device)
        logger.info(f"Evaluating {N} samples across {self.num_depths} depth levels on {self.device}...")

        all_preds = []
        for i in range(N):
            bx = x_tensor[i:i+1].to(self.device)
            bp = self.model(bx, depth_indices)
            all_preds.append(bp.cpu())

        preds = torch.cat(all_preds, dim=0) # [N, 15, 101, 241]
        targets = y_tensor.cpu()           # [N, 15, 101, 241]

        # 1. Depth-wise metrics
        depth_results = {}
        for d_idx, depth_m in enumerate(self.depth_values):
            p_d = preds[:, d_idx, :, :]
            t_d = targets[:, d_idx, :, :]
            m = calculate_metrics(p_d, t_d)
            depth_results[f"{int(depth_m)}m"] = m

        # 2. Overall basin-wide metrics
        overall_m = calculate_metrics(preds, targets)

        # 3. Spatial error distributions: Mean Absolute Error and Bias per grid point
        valid_mask = ~torch.isnan(targets)
        diff = preds - torch.where(valid_mask, targets, torch.zeros_like(targets))
        
        # Mask out land
        diff_masked = torch.where(valid_mask, diff, torch.full_like(diff, float("nan")))
        spatial_mae = torch.nanmean(torch.abs(diff_masked), dim=0).numpy() # [15, 101, 241]
        spatial_bias = torch.nanmean(diff_masked, dim=0).numpy()          # [15, 101, 241]

        summary = {
            "model_type": self.model_type,
            "checkpoint": self.checkpoint_path,
            "num_samples": N,
            "dates": dates or [],
            "overall_metrics": overall_m,
            "depth_metrics": depth_results,
            "spatial_mae_summary": {
                f"{int(d)}m_mean_mae": float(np.nanmean(spatial_mae[i]))
                for i, d in enumerate(self.depth_values)
            }
        }

        self._print_evaluation_table(depth_results, overall_m)
        return summary

    def _print_evaluation_table(self, depth_results: Dict[str, Dict[str, float]], overall: Dict[str, float]):
        """Prints high-visibility scientific evaluation table."""
        print("\n" + "=" * 76)
        print(f"  OCEANEMBED SUBSURFACE TEMPERATURE EVALUATION RESULTS ({self.model_type.upper()})")
        print("=" * 76)
        print(f" {'Depth':>8s} | {'RMSE (°C)':>10s} | {'MAE (°C)':>10s} | {'Bias (°C)':>10s} | {'Pearson r':>10s} ")
        print("-" * 76)
        for depth_str, m in depth_results.items():
            rmse = f"{m['rmse']:.4f}" if not np.isnan(m['rmse']) else "N/A"
            mae = f"{m['mae']:.4f}" if not np.isnan(m['mae']) else "N/A"
            bias = f"{m['bias']:+.4f}" if not np.isnan(m['bias']) else "N/A"
            corr = f"{m['corr']:.4f}" if not np.isnan(m['corr']) else "N/A"
            print(f" {depth_str:>8s} | {rmse:>10s} | {mae:>10s} | {bias:>10s} | {corr:>10s} ")
        print("-" * 76)
        print(f" {'OVERALL':>8s} | {overall['rmse']:>10.4f} | {overall['mae']:>10.4f} | {overall['bias']:>+10.4f} | {overall['corr']:>10.4f} ")
        print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SIH26066 — OceanEmbed Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model", default="oceanembed", choices=["oceanembed", "mlp", "simple_cnn"])
    parser.add_argument("--dataset", default="data/processed/chunk_2020_01_pilot.pt", help="Path to evaluation dataset chunk")
    parser.add_argument("--output-json", default="reports/evaluation_summary.json", help="Path to output JSON summary")

    args = parser.parse_args()

    # Load evaluation chunk
    logger.info(f"Loading evaluation dataset: {args.dataset}")
    raw_pt = torch.load(args.dataset, weights_only=False)
    x = raw_pt["X"]
    y = raw_pt["Y"]
    dates = raw_pt.get("dates", [])

    evaluator = SpatialEvaluator(checkpoint_path=args.checkpoint, model_type=args.model)
    summary = evaluator.evaluate_dataset(x, y, dates)

    # Save report
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Evaluation report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
