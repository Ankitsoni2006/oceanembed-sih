"""
SIH26066 — OceanEmbed FastAPI Backend Service
Provides high-performance RESTful endpoints for 3D subsurface temperature prediction,
oceanographic index computation (MLD, Thermocline, OHC), model evaluation metrics,
independent ARGO in-situ validation reports, and training history telemetry.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.data.catalog import TARGET_GRID, TARGET_DEPTHS, DATA_CATALOG
from src.inference.predict import OceanEmbedPredictor

logger = logging.getLogger("OceanEmbed.API")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="OceanEmbed REST API",
    version="1.0.0",
    description="Deep Learning Subsurface Ocean Temperature Reconstruction (SIH26066)"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor lazy-loaded
PREDICTOR: Optional[OceanEmbedPredictor] = None


def get_predictor() -> OceanEmbedPredictor:
    global PREDICTOR
    if PREDICTOR is None:
        ckpt_path = "checkpoints/oceanembed_best.pt"
        if not os.path.exists(ckpt_path):
            ckpt_path = "checkpoints/oceanembed_latest.pt"
        PREDICTOR = OceanEmbedPredictor(checkpoint_path=ckpt_path)
    return PREDICTOR


class PredictionRequest(BaseModel):
    lat: float = Field(..., ge=5.0, le=30.0, description="Latitude in degrees North (5.0 to 30.0)")
    lon: float = Field(..., ge=45.0, le=105.0, description="Longitude in degrees East (45.0 to 105.0)")
    date: Optional[str] = Field("2020-01-01", description="Date string (YYYY-MM-DD)")


@app.get("/health")
def health_check():
    """System health check and hardware capability report."""
    device_name = "NVIDIA CUDA" if torch.cuda.is_available() else "CPU"
    has_checkpoints = os.path.exists("checkpoints/oceanembed_best.pt") or os.path.exists("checkpoints/oceanembed_latest.pt")
    has_processed = os.path.exists("data/processed/dataset_index.json")

    return {
        "status": "HEALTHY",
        "service": "OceanEmbed Subsurface Reconstruction API",
        "problem_statement": "SIH26066",
        "hardware": {
            "device": device_name,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        },
        "artifacts_available": {
            "model_checkpoints": has_checkpoints,
            "processed_dataset": has_processed,
            "evaluation_report": os.path.exists("reports/evaluation_summary.json"),
            "argo_validation": os.path.exists("reports/argo_validation.json")
        }
    }


@app.get("/catalog")
def get_catalog():
    """Returns grid specification, depth levels, and Copernicus dataset catalog."""
    return {
        "region": "North Indian Ocean (NIO)",
        "spatial_grid": {
            "lat_min": TARGET_GRID.lat_min,
            "lat_max": TARGET_GRID.lat_max,
            "lon_min": TARGET_GRID.lon_min,
            "lon_max": TARGET_GRID.lon_max,
            "resolution_deg": TARGET_GRID.resolution,
            "shape": TARGET_GRID.shape
        },
        "depth_levels_m": list(TARGET_DEPTHS.depths),
        "input_channels": [
            "analysed_sst", "sss", "sla", "uo", "vo", "eastward_wind", "northward_wind",
            "mask_sst", "mask_sss", "mask_sla", "mask_uo", "mask_vo", "mask_wind_u", "mask_wind_v"
        ],
        "datasets": {
            k: {
                "name": v.name,
                "dataset_id": v.dataset_id,
                "variables": v.variables,
                "resolution": v.native_spatial_res_deg,
                "frequency": v.temporal_frequency
            }
            for k, v in DATA_CATALOG.items()
        }
    }


@app.post("/predict")
def predict_subsurface(req: PredictionRequest):
    """
    Reconstructs 15-depth vertical temperature profile at a specific (lat, lon)
    using the trained OceanEmbed deep learning model.
    """
    predictor = get_predictor()
    
    # Map (lat, lon) to nearest grid cell
    lat_idx = int(round((req.lat - TARGET_GRID.lat_min) / TARGET_GRID.resolution))
    lon_idx = int(round((req.lon - TARGET_GRID.lon_min) / TARGET_GRID.resolution))

    # Clamp to grid bounds
    lat_idx = max(0, min(TARGET_GRID.shape[0] - 1, lat_idx))
    lon_idx = max(0, min(TARGET_GRID.shape[1] - 1, lon_idx))

    # Load input sample from processed cache
    chunk_path = "data/processed/chunk_2020_01_pilot.pt"
    if not os.path.exists(chunk_path):
        raise HTTPException(status_code=404, detail="Processed data chunk not found.")

    raw_pt = torch.load(chunk_path, weights_only=False)
    x = raw_pt["X"][0:1] # [1, 14, 101, 241]

    pred_3d = predictor.predict_tensor(x)[0] # [15, 101, 241]
    profile = pred_3d[:, lat_idx, lon_idx].tolist()

    # Derived oceanographic indices
    indices = predictor.compute_oceanographic_indices(pred_3d)
    mld = float(indices["mld_meters"][lat_idx, lon_idx])
    thermocline = float(indices["thermocline_depth_meters"][lat_idx, lon_idx])
    ohc300 = float(indices["ohc300_gj_m2"][lat_idx, lon_idx])

    depth_profile = []
    for d, t in zip(TARGET_DEPTHS.depths, profile):
        depth_profile.append({
            "depth_m": float(d),
            "predicted_temp_c": round(float(t), 3) if not np.isnan(t) else None
        })

    return {
        "query": {
            "lat": req.lat,
            "lon": req.lon,
            "grid_lat": TARGET_GRID.lats[lat_idx],
            "grid_lon": TARGET_GRID.lons[lon_idx],
            "date": req.date
        },
        "profile": depth_profile,
        "oceanographic_indicators": {
            "mixed_layer_depth_m": round(mld, 2) if not np.isnan(mld) else None,
            "thermocline_depth_m": round(thermocline, 2) if not np.isnan(thermocline) else None,
            "ocean_heat_content_300m_gj_m2": round(ohc300, 3) if not np.isnan(ohc300) else None
        }
    }


@app.get("/evaluation")
def get_evaluation():
    """Returns spatial and depth-wise evaluation metrics against GLORYS reference."""
    report_path = "reports/evaluation_summary.json"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Evaluation report has not been generated yet.")
    with open(report_path, "r") as f:
        data = json.load(f)
    return data


@app.get("/argo-validation")
def get_argo_validation():
    """Returns independent validation metrics against in-situ ARGO profiling floats."""
    report_path = "reports/argo_validation.json"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="ARGO validation report has not been generated yet.")
    with open(report_path, "r") as f:
        data = json.load(f)
    return data


@app.get("/training-history")
def get_training_history(model: str = "oceanembed"):
    """Returns epoch-by-epoch loss telemetry from model training."""
    hist_path = f"checkpoints/{model}_history.json"
    if not os.path.exists(hist_path):
        raise HTTPException(status_code=404, detail=f"Training history for {model} not found.")
    with open(hist_path, "r") as f:
        data = json.load(f)
    return data
