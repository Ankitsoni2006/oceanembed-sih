"""
SIH26066 — OceanEmbed Subsurface Temperature Inference Engine
Performs model inference to reconstruct 3D temperature fields (0–1000m)
from 14-channel surface observations and derives physical oceanographic parameters
including Mixed Layer Depth (MLD), Thermocline Depth, and Ocean Heat Content (OHC300).
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import xarray as xr
import torch

from src.models.oceanembed import OceanEmbedNet
from src.data.catalog import TARGET_GRID, TARGET_DEPTHS
from src.preprocessing.normalization import OceanStandardScaler

logger = logging.getLogger("OceanEmbed.Inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class OceanEmbedPredictor:
    """
    High-level predictor serving reconstructed 3D subsurface ocean temperature
    and physical oceanographic indicators.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/oceanembed_best.pt",
        device: Optional[torch.device] = None,
        depth_values: Optional[List[float]] = None
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.depth_values = np.array(depth_values or list(TARGET_DEPTHS.depths), dtype=np.float32)
        self.num_depths = len(self.depth_values)
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model()

    def _load_model(self) -> OceanEmbedNet:
        logger.info(f"Loading OceanEmbedNet from {self.checkpoint_path} on {self.device}...")
        model = OceanEmbedNet(
            in_vars=7,
            num_depths=self.num_depths,
            base_features=32,
            embedding_dim=128
        )
        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            logger.info("Successfully loaded model checkpoint weights.")
        else:
            logger.warning(f"Checkpoint {self.checkpoint_path} not found. Running with randomly initialized weights.")
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict_tensor(self, x: torch.Tensor) -> np.ndarray:
        """
        Runs forward inference.
        x: [B, 14, 101, 241] or [14, 101, 241]
        Returns: [B, 15, 101, 241] numpy array of temperatures in °C
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        depth_indices = torch.arange(self.num_depths, device=self.device)
        pred = self.model(x, depth_indices)
        return pred.cpu().numpy()

    def compute_oceanographic_indices(self, temp_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes Mixed Layer Depth (MLD), Thermocline Depth, and Ocean Heat Content (OHC300).
        temp_3d: [15, 101, 241] in °C
        """
        depths = self.depth_values
        H, W = temp_3d.shape[1], temp_3d.shape[2]

        mld = np.full((H, W), np.nan, dtype=np.float32)
        thermocline_depth = np.full((H, W), np.nan, dtype=np.float32)
        ohc_300 = np.full((H, W), np.nan, dtype=np.float32)

        # Standard physical constants for seawater
        rho_0 = 1025.0  # kg/m^3
        c_p = 3990.0    # J / (kg * °C)

        idx_300 = np.where(depths <= 300.0)[0]
        depths_300 = depths[idx_300]

        for i in range(H):
            for j in range(W):
                t_col = temp_3d[:, i, j]
                if np.isnan(t_col[0]):
                    continue

                sst = t_col[0]

                # 1. Mixed Layer Depth (MLD): depth where T drops by 0.2°C from SST
                drop_idx = np.where(t_col <= (sst - 0.2))[0]
                if len(drop_idx) > 0:
                    first_drop = drop_idx[0]
                    if first_drop == 0:
                        mld[i, j] = depths[0]
                    else:
                        # Linear interpolation between depths
                        t0 = t_col[first_drop - 1]
                        t1 = t_col[first_drop]
                        z0 = depths[first_drop - 1]
                        z1 = depths[first_drop]
                        if t0 != t1:
                            mld[i, j] = z0 + (sst - 0.2 - t0) * (z1 - z0) / (t1 - t0)
                        else:
                            mld[i, j] = z0
                else:
                    mld[i, j] = depths[-1]

                # 2. Thermocline Depth: location of maximum negative temperature gradient (-dT/dz)
                # Compute discrete gradients between consecutive depth levels
                grad = -(t_col[1:] - t_col[:-1]) / (depths[1:] - depths[:-1] + 1e-6)
                if len(grad) > 0 and not np.all(np.isnan(grad)):
                    max_grad_idx = int(np.nanargmax(grad))
                    thermocline_depth[i, j] = (depths[max_grad_idx] + depths[max_grad_idx + 1]) / 2.0

                # 3. Ocean Heat Content (OHC300): rho * cp * integral(T dz) from 0 to 300m
                t_300 = t_col[idx_300]
                if not np.any(np.isnan(t_300)):
                    integrated_t = np.trapezoid(t_300, depths_300)
                    # Result in Gigajoules per square meter (GJ/m^2)
                    ohc_300[i, j] = (rho_0 * c_p * integrated_t) / 1e9

        return {
            "mld_meters": mld,
            "thermocline_depth_meters": thermocline_depth,
            "ohc300_gj_m2": ohc_300
        }

    def export_to_netcdf(
        self,
        temp_3d: np.ndarray,
        output_filepath: str,
        date_str: str = "2020-01-01",
        extra_vars: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Exports reconstructed 3D temperature and derived indicators to NetCDF4.
        """
        lats = TARGET_GRID.lats
        lons = TARGET_GRID.lons
        depths = self.depth_values

        data_vars = {
            "thetao_pred": (["depth", "lat", "lon"], temp_3d, {
                "long_name": "OceanEmbed Reconstructed Potential Temperature",
                "units": "degrees_C",
                "standard_name": "sea_water_potential_temperature"
            })
        }

        if extra_vars:
            for name, arr in extra_vars.items():
                data_vars[name] = (["lat", "lon"], arr, {"long_name": name})

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "depth": ("depth", depths, {"units": "meters", "positive": "down"}),
                "lat": ("lat", lats, {"units": "degrees_north"}),
                "lon": ("lon", lons, {"units": "degrees_east"}),
                "time": [np.datetime64(date_str)]
            },
            attrs={
                "title": "OceanEmbed 3D Subsurface Ocean Temperature Reconstruction",
                "project": "SIH26066",
                "institution": "Smart India Hackathon 2026",
                "source": f"OceanEmbedNet ({self.checkpoint_path})"
            }
        )

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        ds.to_netcdf(output_filepath)
        logger.info(f"Exported 3D reconstruction NetCDF to: {output_filepath} ({os.path.getsize(output_filepath):,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="SIH26066 — OceanEmbed Inference")
    parser.add_argument("--checkpoint", default="checkpoints/oceanembed_best.pt", help="Path to checkpoint")
    parser.add_argument("--dataset", default="data/processed/chunk_2020_01_pilot.pt", help="Path to input tensor chunk")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index within chunk")
    parser.add_argument("--output-nc", default="data/processed/prediction_output.nc", help="Output NetCDF path")

    args = parser.parse_args()

    raw_pt = torch.load(args.dataset, weights_only=False)
    x = raw_pt["X"][args.sample_idx]
    date_str = raw_pt.get("dates", ["2020-01-01"])[args.sample_idx]

    predictor = OceanEmbedPredictor(checkpoint_path=args.checkpoint)
    pred_3d = predictor.predict_tensor(x)[0]  # [15, 101, 241]

    # Compute physical indicators
    logger.info("Computing MLD, Thermocline Depth, and Ocean Heat Content...")
    indices = predictor.compute_oceanographic_indices(pred_3d)

    predictor.export_to_netcdf(pred_3d, args.output_nc, date_str=date_str, extra_vars=indices)
    print(f"\nInference complete! Output saved to: {args.output_nc}")


if __name__ == "__main__":
    main()
