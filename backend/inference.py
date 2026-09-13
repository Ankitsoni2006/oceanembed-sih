"""
SIH26066 — OceanEmbed Backend Inference Engine
Manages model loading, zero-leakage preprocessing, grid mapping, and real-time 3D reconstruction.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from backend.config import (
    PROJECT_ROOT,
    CHECKPOINT_PATH,
    SCALER_PATH,
    PROCESSED_DATA_DIR,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    GRID_RESOLUTION,
    N_LATS,
    N_LONS,
    TARGET_DEPTHS,
    INPUT_VARIABLES,
    MODEL_NAME,
    MODEL_VERSION,
    MODEL_PARAMETERS
)

# Ensure project root is in sys.path so model and preprocessing modules are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.oceanembed_v3_decoder import OceanEmbedNetV3_Decoder
from src.preprocessing.normalization import OceanStandardScaler

logger = logging.getLogger("OceanEmbed.Backend.Inference")


class OceanEmbedInferenceService:
    """
    Singleton service that maintains the loaded OceanEmbedNetV3_Decoder model,
    the fitted training scaler, and an in-memory date index for real-time inference.
    """

    def __init__(self):
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[OceanEmbedNetV3_Decoder] = None
        self.scaler: Optional[OceanStandardScaler] = None
        self.date_to_chunk_index: Dict[str, Tuple[Path, int]] = {}
        self.available_dates: List[str] = []
        # In-memory chunk cache (LRU up to 3 chunks ~250MB RAM)
        self.chunk_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.max_cached_chunks: int = 3
        self.is_initialized: bool = False

    def initialize(self):
        """
        Executes startup loading once:
        1. Instantiates OceanEmbedNetV3_Decoder
        2. Loads weights from checkpoint
        3. Sets model.eval() and device
        4. Loads the validated training scaler
        5. Indexes available processed dates
        """
        if self.is_initialized:
            logger.info("OceanEmbedInferenceService is already initialized.")
            return

        logger.info(f"Initializing OceanEmbedInferenceService on device: {self.device}")

        # 1. Load Model Checkpoint
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {CHECKPOINT_PATH}")

        logger.info(f"Loading model weights from {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}...")
        self.model = OceanEmbedNetV3_Decoder(
            in_vars=7,
            num_depths=len(TARGET_DEPTHS),
            base_features=32,
            embedding_dim=128
        )
        ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Warm up model with dummy forward pass under inference_mode
        dummy_x = torch.zeros(1, 14, N_LATS, N_LONS, device=self.device)
        with torch.inference_mode():
            _ = self.model(dummy_x)
        logger.info("Model warm-up completed successfully.")

        # 2. Load Scaler
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler parameters not found at: {SCALER_PATH}")

        logger.info(f"Loading standard scaler from {SCALER_PATH.relative_to(PROJECT_ROOT)}...")
        self.scaler = OceanStandardScaler.load(str(SCALER_PATH))
        logger.info("Scaler loaded successfully.")

        # 3. Index Available Dates from Processed Chunks (chunk_2020_01 to chunk_2020_09)
        self._build_date_index()

        self.is_initialized = True
        logger.info(f"OceanEmbedInferenceService initialization complete. {len(self.available_dates)} dates available.")

    def _build_date_index(self):
        """
        Scans monthly processed chunks and maps each YYYY-MM-DD date to its chunk file and day offset.
        """
        logger.info(f"Scanning processed data directory: {PROCESSED_DATA_DIR.relative_to(PROJECT_ROOT)}...")
        # Target standard monthly chunks in order
        monthly_chunks = [f"chunk_2020_{m:02d}.pt" for m in range(1, 10)]
        for chunk_name in monthly_chunks:
            chunk_path = PROCESSED_DATA_DIR / chunk_name
            if chunk_path.exists():
                data = torch.load(chunk_path, weights_only=False)
                dates = data.get("dates", [])
                for offset, dt in enumerate(dates):
                    self.date_to_chunk_index[dt] = (chunk_path, offset)

        self.available_dates = sorted(list(self.date_to_chunk_index.keys()))
        logger.info(f"Indexed {len(self.available_dates)} dates ({self.available_dates[0]} to {self.available_dates[-1]}).")

    def _get_day_tensor(self, date_str: str) -> torch.Tensor:
        """
        Retrieves the 14-channel surface tensor [14, 101, 241] for a specified date,
        utilizing an LRU chunk memory cache.
        """
        if date_str not in self.date_to_chunk_index:
            raise KeyError(f"Date '{date_str}' is not available in the processed 2020 dataset.")

        chunk_path, offset = self.date_to_chunk_index[date_str]
        chunk_key = str(chunk_path)

        if chunk_key in self.chunk_cache:
            # Move to end (most recently used)
            self.chunk_cache.move_to_end(chunk_key)
            x_chunk = self.chunk_cache[chunk_key]
        else:
            # Load chunk into cache
            logger.debug(f"Loading chunk {chunk_path.name} into memory cache...")
            chunk_data = torch.load(chunk_path, weights_only=False)
            x_chunk = chunk_data["X"]  # [N_days, 14, 101, 241]
            self.chunk_cache[chunk_key] = x_chunk
            if len(self.chunk_cache) > self.max_cached_chunks:
                # Evict oldest
                self.chunk_cache.popitem(last=False)

        return x_chunk[offset]  # [14, 101, 241]

    def map_coordinates_to_grid(self, lat: float, lon: float) -> Tuple[int, int, float, float]:
        """
        Maps continuous (latitude, longitude) coordinates to nearest 0.25° grid indices.
        Returns: (lat_idx, lon_idx, grid_lat, grid_lon)
        """
        lat_idx = int(round((lat - LAT_MIN) / GRID_RESOLUTION))
        lon_idx = int(round((lon - LON_MIN) / GRID_RESOLUTION))

        # Clamp strictly within bounds [0, 100] and [0, 240]
        lat_idx = max(0, min(N_LATS - 1, lat_idx))
        lon_idx = max(0, min(N_LONS - 1, lon_idx))

        grid_lat = round(LAT_MIN + lat_idx * GRID_RESOLUTION, 4)
        grid_lon = round(LON_MIN + lon_idx * GRID_RESOLUTION, 4)

        return lat_idx, lon_idx, grid_lat, grid_lon

    @staticmethod
    def compute_oceanographic_indices(depths: List[int], temps: List[float]) -> Dict[str, Optional[float]]:
        """
        Computes standard physical oceanographic indicators from reconstructed 15-depth column:
        1. Mixed Layer Depth (MLD): depth where T drops by >= 0.2°C from surface SST.
        2. Thermocline Depth: depth of maximum vertical temperature gradient max |dT/dz|.
        3. Upper Ocean Heat Content (OHC300): integrated heat content 0–300m in GJ/m².
        """
        if len(temps) != len(depths) or any(np.isnan(t) for t in temps):
            return {"mixed_layer_depth_m": None, "thermocline_depth_m": None, "ocean_heat_content_300m_gj_m2": None}

        z = np.array(depths, dtype=np.float32)
        t = np.array(temps, dtype=np.float32)
        sst = t[0]

        # 1. MLD: 0.2°C threshold criterion
        mld = None
        for i in range(1, len(z)):
            if (sst - t[i]) >= 0.2:
                # Linear interpolation to exact 0.2°C crossing
                dT = t[i - 1] - t[i]
                if abs(dT) > 1e-5:
                    frac = ((sst - 0.2) - t[i]) / (t[i - 1] - t[i])
                    mld = float(z[i] - frac * (z[i] - z[i - 1]))
                else:
                    mld = float(z[i])
                break
        if mld is None:
            mld = float(z[-1])

        # 2. Thermocline Depth: max vertical gradient |dT/dz|
        dz = np.diff(z)
        dt = np.diff(t)
        gradient = np.abs(dt / np.maximum(dz, 1e-4))
        max_grad_idx = int(np.argmax(gradient))
        # Midpoint of the steepest interval
        thermocline_depth = float((z[max_grad_idx] + z[max_grad_idx + 1]) / 2.0)

        # 3. OHC300: rho * cp * integral(T dz) from 0 to 300m
        rho_0 = 1025.0  # kg/m³
        c_p = 3990.0    # J / (kg * °C)
        idx_300 = np.where(z <= 300)[0]
        z_sub = z[idx_300]
        t_sub = t[idx_300]
        ohc_joules = float(np.trapezoid(t_sub, z_sub) * rho_0 * c_p)
        ohc_gj = float(ohc_joules / 1e9)  # Convert to GJ/m²

        return {
            "mixed_layer_depth_m": round(mld, 2),
            "thermocline_depth_m": round(thermocline_depth, 2),
            "ocean_heat_content_300m_gj_m2": round(ohc_gj, 3)
        }

    def predict(self, date_str: str, lat: float, lon: float) -> Dict[str, Any]:
        """
        Executes end-to-end real inference for a target date and coordinate.
        """
        t_start = time.perf_counter()

        if not self.is_initialized:
            self.initialize()

        # 1. Date existence validation
        if date_str not in self.date_to_chunk_index:
            raise KeyError(
                f"Date '{date_str}' is unavailable. Available date range: "
                f"{self.available_dates[0]} to {self.available_dates[-1]} ({len(self.available_dates)} days total)."
            )

        # 2. Coordinate mapping
        lat_idx, lon_idx, grid_lat, grid_lon = self.map_coordinates_to_grid(lat, lon)

        # 3. Retrieve raw 14-channel surface observation tensor
        x_day = self._get_day_tensor(date_str)  # [14, 101, 241]

        # 4. Ocean validity check: Channel 7 is mask_sst (1.0 = ocean water, 0.0 = land/missing)
        is_ocean = bool(float(x_day[7, lat_idx, lon_idx].item()) > 0.5)
        if not is_ocean:
            return {
                "date": date_str,
                "latitude": lat,
                "longitude": lon,
                "grid_latitude": grid_lat,
                "grid_longitude": grid_lon,
                "is_valid_ocean": False,
                "error": "Land or invalid ocean cell",
                "message": f"Requested coordinates ({lat}°N, {lon}°E) mapped to grid cell ({grid_lat}°N, {grid_lon}°E) which is on land or has missing satellite observations."
            }

        # 5. Extract raw surface observation values for metadata
        surface_obs = {
            "sst_c": round(float(x_day[0, lat_idx, lon_idx].item()), 3),
            "sss_psu": round(float(x_day[1, lat_idx, lon_idx].item()), 3),
            "ssh_m": round(float(x_day[2, lat_idx, lon_idx].item()), 3),
            "u_current_ms": round(float(x_day[3, lat_idx, lon_idx].item()), 3),
            "v_current_ms": round(float(x_day[4, lat_idx, lon_idx].item()), 3),
            "u_wind_ms": round(float(x_day[5, lat_idx, lon_idx].item()), 3),
            "v_wind_ms": round(float(x_day[6, lat_idx, lon_idx].item()), 3),
        }

        # 6. Apply validated zero-leakage standard scaler
        # Shape: [1, 14, 101, 241]
        x_scaled = self.scaler.transform(x_day.unsqueeze(0)).to(self.device)

        # 7. Model forward inference under torch.inference_mode()
        t_inf0 = time.perf_counter()
        with torch.inference_mode():
            pred_3d = self.model(x_scaled)  # [1, 15, 101, 241]
        t_inf_ms = (time.perf_counter() - t_inf0) * 1000.0

        # 8. Extract 15-depth vertical temperature column at target grid cell
        pred_col = pred_3d[0, :, lat_idx, lon_idx].cpu().numpy()
        temps_c = [round(float(v), 4) for v in pred_col]

        # 9. Compute oceanographic indicators
        indicators = self.compute_oceanographic_indices(TARGET_DEPTHS, temps_c)

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "date": date_str,
            "latitude": lat,
            "longitude": lon,
            "grid_latitude": grid_lat,
            "grid_longitude": grid_lon,
            "depths_m": TARGET_DEPTHS,
            "temperatures_c": temps_c,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "inference_ms": round(t_inf_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "is_valid_ocean": True,
            "surface_observations": surface_obs,
            "oceanographic_indicators": indicators
        }


# Global service instance
INFERENCE_SERVICE = OceanEmbedInferenceService()
