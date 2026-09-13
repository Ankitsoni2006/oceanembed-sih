"""
SIH26066 — OceanEmbed Backend Configuration
Centralizes filesystem paths, spatial grid constants, depth levels, and service parameters.
"""

import os
from pathlib import Path
from typing import List

# Determine Project Root robustly
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model Checkpoint & Scaler Paths
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "phase5" / "oceanembed_v3_decoder.pt"
SCALER_PATH = PROJECT_ROOT / "configs" / "scaler_params_experiment_2020.json"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Spatial Grid Configuration (0.25° × 0.25° North Indian Ocean)
LAT_MIN: float = 5.0
LAT_MAX: float = 30.0
LON_MIN: float = 45.0
LON_MAX: float = 105.0
GRID_RESOLUTION: float = 0.25
N_LATS: int = 101  # (30.0 - 5.0) / 0.25 + 1
N_LONS: int = 241  # (105.0 - 45.0) / 0.25 + 1

# Standard 15 Target Depth Levels (meters)
TARGET_DEPTHS: List[int] = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# Input Variables (7 Physical Surface Variables)
INPUT_VARIABLES: List[str] = [
    "SST",
    "SSS",
    "SSH",
    "surface_u_current",
    "surface_v_current",
    "surface_u_wind",
    "surface_v_wind"
]

# Model Metadata
MODEL_NAME: str = "OceanEmbedNetV3_Decoder"
MODEL_VERSION: str = "v3"
MODEL_PARAMETERS: int = 1275934
MODEL_STATUS: str = "validated"

# CORS Configuration for Frontend Development & Network Demo
ALLOWED_ORIGINS: List[str] = ["*"]
