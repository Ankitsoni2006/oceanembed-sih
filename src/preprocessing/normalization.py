"""
SIH26066 — OceanEmbed Feature Normalization Module
Standardizes physical surface features using train-only statistics with strict land-zeroing.
Preserves binary mask channels without distortion.
"""

import json
import os
from typing import Dict, List, Optional, Union
import numpy as np
import torch


class OceanStandardScaler:
    """
    Z-score standard scaler designed for multi-channel oceanic tensor data.
    Computes statistics strictly over valid ocean pixels from the training partition.
    Leaves binary mask channels un-normalized.
    """
    def __init__(self, num_physical_channels: int = 7, eps: float = 1e-6):
        self.num_physical_channels = num_physical_channels
        self.eps = eps
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, x_data: Union[torch.Tensor, np.ndarray]) -> "OceanStandardScaler":
        """
        Fits mean and standard deviation per physical channel using valid ocean pixels only.
        x_data shape: [N, 14, H, W] or [14, H, W]
        Channels 0..6 are physical variables; channels 7..13 are corresponding binary masks.
        """
        if isinstance(x_data, torch.Tensor):
            x_np = x_data.detach().cpu().numpy()
        else:
            x_np = np.array(x_data)

        if x_np.ndim == 3:
            x_np = np.expand_dims(x_np, axis=0)  # [1, 14, H, W]

        means = []
        stds = []

        for c in range(self.num_physical_channels):
            var_vals = x_np[:, c]
            mask_vals = x_np[:, c + self.num_physical_channels]
            
            # Select strictly valid ocean pixels
            valid_mask = (mask_vals == 1.0) & (~np.isnan(var_vals)) & (~np.isinf(var_vals))
            valid_pixels = var_vals[valid_mask]

            if len(valid_pixels) == 0:
                raise ValueError(f"Channel {c} has zero valid ocean pixels to fit scaler!")

            mu = float(valid_pixels.mean())
            sigma = float(valid_pixels.std())
            if sigma < self.eps:
                sigma = 1.0  # Avoid zero division on constant fields

            means.append(mu)
            stds.append(sigma)

        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)
        self.is_fitted = True
        return self

    def transform(self, x_data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Transforms physical channels using fitted (mu, sigma).
        Zeroes out all masked / unobserved pixels.
        Leaves mask channels untouched.
        """
        if not self.is_fitted:
            raise RuntimeError("OceanStandardScaler must be fitted before transforming data!")

        is_torch = isinstance(x_data, torch.Tensor)
        if is_torch:
            orig_device = x_data.device
            orig_dtype = x_data.dtype
            x_np = x_data.detach().cpu().numpy().copy()
        else:
            x_np = np.array(x_data, copy=True)

        squeeze_output = False
        if x_np.ndim == 3:
            x_np = np.expand_dims(x_np, axis=0)
            squeeze_output = True

        for c in range(self.num_physical_channels):
            mu = self.means[c]
            sigma = self.stds[c]
            mask = x_np[:, c + self.num_physical_channels]

            # Standardize
            x_np[:, c] = (x_np[:, c] - mu) / (sigma + self.eps)
            # Zero out land / missing / invalid pixels
            x_np[:, c] = np.where(mask == 1.0, x_np[:, c], 0.0)

        if squeeze_output:
            x_np = x_np[0]

        if is_torch:
            return torch.from_numpy(x_np).to(device=orig_device, dtype=orig_dtype)
        return x_np

    def fit_transform(self, x_data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        return self.fit(x_data).transform(x_data)

    def save(self, filepath: str):
        """Saves fitted normalization parameters to a JSON file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler parameters!")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        meta = {
            "num_physical_channels": self.num_physical_channels,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "eps": self.eps
        }
        with open(filepath, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "OceanStandardScaler":
        """Loads fitted normalization parameters from a JSON file."""
        with open(filepath, "r") as f:
            meta = json.load(f)
        
        num_ch = meta.get("num_physical_channels")
        if num_ch is None:
            if "channels" in meta:
                num_ch = len(meta["channels"])
            elif "means_array" in meta:
                num_ch = len(meta["means_array"])
            elif isinstance(meta.get("means"), list):
                num_ch = len(meta["means"])
            else:
                num_ch = 7

        scaler = cls(num_physical_channels=num_ch, eps=meta.get("eps", 1e-6))

        if "means_array" in meta:
            means = meta["means_array"]
        elif isinstance(meta.get("means"), dict):
            means = list(meta["means"].values())
        else:
            means = meta["means"]

        if "stds_array" in meta:
            stds = meta["stds_array"]
        elif isinstance(meta.get("stds"), dict):
            stds = list(meta["stds"].values())
        else:
            stds = meta["stds"]

        scaler.means = np.array(means, dtype=np.float32)
        scaler.stds = np.array(stds, dtype=np.float32)
        scaler.is_fitted = True
        return scaler
