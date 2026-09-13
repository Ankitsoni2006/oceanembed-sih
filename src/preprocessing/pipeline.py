"""
SIH26066 — OceanEmbed End-to-End Data Pipeline Engine
Orchestrates raw fetching, temporal alignment, horizontal regridding,
vertical interpolation, 14-channel masking, and quality control.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import xarray as xr
import torch

from src.data.catalog import TARGET_GRID, TARGET_DEPTHS, DATA_CATALOG
from src.data.acquisition import CopernicusAcquisitionEngine
from src.preprocessing.grid import regrid_to_target_grid
from src.preprocessing.temporal import (
    extract_exact_daily_slice,
    aggregate_hourly_winds_to_daily,
    interpolate_weekly_sss_to_daily,
    TemporalAlignmentError
)
from src.preprocessing.glorys import process_glorys_3d_slice
from src.preprocessing.masks import concatenate_variables_and_masks

logger = logging.getLogger("OceanEmbed.Pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class QualityControlError(Exception):
    """Raised when processed ocean tensor violates physical or structural QC."""
    pass


class OceanDataPipeline:
    """
    Unified multi-year capable pipeline building aligned (X, Y) daily samples.
    """
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.acquisition = CopernicusAcquisitionEngine(raw_data_dir=self.raw_dir)

    def process_day(self, target_date: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Executes the complete processing sequence for a single calendar day.
        Returns:
            X: [1, 14, 101, 241] (7 variables in physical units + 7 validity masks)
            Y: [1, 15, 101, 241] (15 subsurface depth temperatures in °C)
            manifest: Metadata dictionary documenting sources, provenance, QC stats
        """
        logger.info(f"--- Processing target date: {target_date} ---")
        t_start = datetime.now()

        # 1. Fetch raw files
        f_sst = self.acquisition.fetch_dataset_slice("SST", target_date, target_date)
        f_sss = self.acquisition.fetch_dataset_slice("SSS", target_date, target_date)
        f_ssh = self.acquisition.fetch_dataset_slice("SSH", target_date, target_date)
        f_cur = self.acquisition.fetch_dataset_slice("CURRENTS", target_date, target_date)
        f_wnd = self.acquisition.fetch_dataset_slice("WINDS", target_date, target_date)
        f_glo = self.acquisition.fetch_dataset_slice("GLORYS", target_date, target_date)

        # 2. Open and temporally synchronize
        with xr.open_dataset(f_sst) as ds_sst_raw:
            ds_sst = extract_exact_daily_slice(ds_sst_raw, target_date)
            # Regrid SST (0.05° -> 0.25°)
            ds_sst_rg = regrid_to_target_grid(ds_sst)
            sst_k = ds_sst_rg["analysed_sst"].values
            sst_c = sst_k - 273.15  # Convert Kelvin to Celsius

        with xr.open_dataset(f_sss) as ds_sss_raw:
            ds_sss, sss_provenance = interpolate_weekly_sss_to_daily(ds_sss_raw, target_date)
            ds_sss_rg = regrid_to_target_grid(ds_sss)
            sss_val = ds_sss_rg["sss"].values
            if sss_val.ndim == 3:
                sss_val = sss_val[0]

        with xr.open_dataset(f_ssh) as ds_ssh_raw:
            ds_ssh = extract_exact_daily_slice(ds_ssh_raw, target_date)
            ds_ssh_rg = regrid_to_target_grid(ds_ssh)
            ssh_val = ds_ssh_rg["sla"].values
            if ssh_val.ndim == 3:
                ssh_val = ssh_val[0]

        with xr.open_dataset(f_cur) as ds_cur_raw:
            ds_cur = extract_exact_daily_slice(ds_cur_raw, target_date)
            if "depth" in ds_cur.coords or "depth" in ds_cur.dims:
                ds_cur = ds_cur.isel(depth=0)
            ds_cur_rg = regrid_to_target_grid(ds_cur)
            uo_val = ds_cur_rg["uo"].values
            vo_val = ds_cur_rg["vo"].values
            if uo_val.ndim == 3:
                uo_val, vo_val = uo_val[0], vo_val[0]

        with xr.open_dataset(f_wnd) as ds_wnd_raw:
            ds_wnd = aggregate_hourly_winds_to_daily(ds_wnd_raw, target_date)
            ds_wnd_rg = regrid_to_target_grid(ds_wnd)
            u_wind = ds_wnd_rg["eastward_wind"].values
            v_wind = ds_wnd_rg["northward_wind"].values
            if u_wind.ndim == 3:
                u_wind, v_wind = u_wind[0], v_wind[0]

        # 3. Process 3D GLORYS Target
        with xr.open_dataset(f_glo) as ds_glo_raw:
            ds_glo_day = extract_exact_daily_slice(ds_glo_raw, target_date)
            ds_glo_target = process_glorys_3d_slice(ds_glo_day)
            y_np = ds_glo_target["thetao"].values  # Shape: (15, 101, 241)
            if y_np.ndim == 4:
                y_np = y_np[0]

        # 4. Assemble 7 surface variables
        if sst_c.ndim == 3:
            sst_c = sst_c[0]

        raw_surface_7 = np.stack([sst_c, sss_val, ssh_val, uo_val, vo_val, u_wind, v_wind], axis=0)  # (7, 101, 241)
        raw_tensor_7 = torch.from_numpy(raw_surface_7).unsqueeze(0).float()  # [1, 7, 101, 241]

        # 5. Generate masks and concatenate -> 14 channels
        X = concatenate_variables_and_masks(raw_tensor_7, missing_val_threshold=-100.0)
        Y = torch.from_numpy(y_np).unsqueeze(0).float()  # [1, 15, 101, 241]

        # 6. Quality Control Assertions
        self._validate_quality(X, Y, target_date)

        elapsed = (datetime.now() - t_start).total_seconds()
        logger.info(f"Successfully processed {target_date} in {elapsed:.2f}s")

        manifest = {
            "date": target_date,
            "processed_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "X_shape": list(X.shape),
            "Y_shape": list(Y.shape),
            "sss_provenance": sss_provenance,
            "qc_passed": True
        }

        return X, Y, manifest

    def _validate_quality(self, X: torch.Tensor, Y: torch.Tensor, date_str: str):
        """Enforces rigorous physical and structural QC on processed tensors."""
        # 1. Structural dimensions
        if X.shape != (1, 14, 101, 241):
            raise QualityControlError(f"[{date_str}] X shape mismatch: expected (1, 14, 101, 241), got {tuple(X.shape)}")
        if Y.shape != (1, 15, 101, 241):
            raise QualityControlError(f"[{date_str}] Y shape mismatch: expected (1, 15, 101, 241), got {tuple(Y.shape)}")

        # 2. NaNs in X
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise QualityControlError(f"[{date_str}] NaNs or Infs leaked into input tensor X!")

        # 3. Masks binary integrity
        masks = X[0, 7:14]
        if not ((masks == 0.0) | (masks == 1.0)).all():
            raise QualityControlError(f"[{date_str}] Non-binary values detected in mask channels!")

        # 4. Target validity
        valid_y_surf = ~torch.isnan(Y[0, 0])
        ocean_cells = valid_y_surf.sum().item()
        if ocean_cells < 10000 or ocean_cells > 13000:
            raise QualityControlError(f"[{date_str}] Anomalous surface ocean cell count: {ocean_cells}")

        # 5. Physical range sanity checks over valid cells
        sst_ocean = X[0, 0][X[0, 7] == 1.0]
        if sst_ocean.min() < 0.0 or sst_ocean.max() > 38.0:
            raise QualityControlError(f"[{date_str}] Unphysical SST: min={sst_ocean.min():.1f}, max={sst_ocean.max():.1f}")
