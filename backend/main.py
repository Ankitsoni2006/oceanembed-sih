"""
SIH26066 — OceanEmbed FastAPI Application Entry Point
Exposes high-performance RESTful endpoints for 3D subsurface temperature reconstruction.
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path regardless of execution directory
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import (
    CHECKPOINT_PATH,
    PROJECT_ROOT,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    GRID_RESOLUTION,
    TARGET_DEPTHS,
    INPUT_VARIABLES,
    MODEL_NAME,
    MODEL_VERSION,
    MODEL_PARAMETERS,
    MODEL_STATUS,
    ALLOWED_ORIGINS
)
from backend.schemas import (
    HealthResponse,
    ModelInfoResponse,
    RegionInfo,
    AvailableDatesResponse,
    DateRange,
    PredictionRequest,
    PredictionResponse,
    ErrorResponse
)
from backend.inference import INFERENCE_SERVICE

# Structured Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("OceanEmbed.API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads model checkpoint, scaler, and date index ONCE at startup.
    """
    logger.info("=" * 70)
    logger.info("STARTING OCEANEMBED V3 FASTAPI BACKEND (SIH26066)")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Active Checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    logger.info("=" * 70)

    try:
        INFERENCE_SERVICE.initialize()
        logger.info(f"Ready for inference on device: {INFERENCE_SERVICE.device}")
    except Exception as e:
        logger.error(f"Fatal error during model initialization: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down OceanEmbed backend service.")


# FastAPI Application Instance
app = FastAPI(
    title="OceanEmbed Subsurface Temperature Reconstruction API",
    version="3.0.0",
    description=(
        "Production-grade FastAPI service serving real 3D subsurface ocean temperature "
        "reconstructions (0–1000m) from surface satellite observations in the North Indian Ocean. "
        "Powered by the validated OceanEmbedNetV3_Decoder architecture (SIH26066)."
    ),
    lifespan=lifespan
)

# Configure CORS for Frontend Integration (Local & Network)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health & Device Status",
    tags=["System"]
)
def get_health() -> HealthResponse:
    """
    Returns system health, whether model weights are resident in memory, and active compute device.
    """
    return HealthResponse(
        status="ok",
        model_loaded=INFERENCE_SERVICE.is_initialized and (INFERENCE_SERVICE.model is not None),
        device=str(INFERENCE_SERVICE.device),
        model=MODEL_NAME
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Architecture & Metadata",
    tags=["Model Telemetry"]
)
def get_model_info() -> ModelInfoResponse:
    """
    Returns verified model architecture specification, parameter count, input channels,
    target depth levels, and domain coordinates.
    """
    rel_ckpt = str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return ModelInfoResponse(
        model=MODEL_NAME,
        version=MODEL_VERSION,
        parameters=MODEL_PARAMETERS,
        input_variables=INPUT_VARIABLES,
        output_depths_m=TARGET_DEPTHS,
        grid_resolution=f"{GRID_RESOLUTION}°",
        region=RegionInfo(
            lat_min=LAT_MIN,
            lat_max=LAT_MAX,
            lon_min=LON_MIN,
            lon_max=LON_MAX
        ),
        device=str(INFERENCE_SERVICE.device),
        checkpoint=rel_ckpt,
        status=MODEL_STATUS
    )


@app.get(
    "/available-dates",
    response_model=AvailableDatesResponse,
    summary="List Supported Observation Dates",
    tags=["Data Catalog"]
)
def get_available_dates(
    format: Optional[str] = Query(None, description="Optional 'list' format returns raw array of date strings")
):
    """
    Returns the dates actually available in the 2020 processed surface-input dataset.
    Arbitrary dates outside the processed archive are not supported.
    """
    if not INFERENCE_SERVICE.is_initialized:
        INFERENCE_SERVICE.initialize()

    dates = INFERENCE_SERVICE.available_dates
    if not dates:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processed data archive is empty or currently unindexed."
        )

    if format == "list":
        return JSONResponse(content=dates)

    return AvailableDatesResponse(
        total_dates=len(dates),
        date_range=DateRange(start=dates[0], end=dates[-1]),
        dates=dates
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query coordinates, land cell, or missing data"},
        404: {"model": ErrorResponse, "description": "Target date unavailable in dataset"},
        422: {"description": "Schema validation failure"}
    },
    summary="Predict 15-Depth Subsurface Temperature Profile",
    tags=["Inference"]
)
def predict_temperature_profile(request: PredictionRequest) -> PredictionResponse:
    """
    Reconstructs the 15-depth vertical ocean temperature profile at the requested coordinate
    using the validated OceanEmbedNetV3_Decoder model.
    """
    # 1. Validate Coordinate Boundaries
    if not (LAT_MIN <= request.latitude <= LAT_MAX):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Latitude {request.latitude}°N is out of bounds. Supported range is {LAT_MIN}°N to {LAT_MAX}°N."
        )
    if not (LON_MIN <= request.longitude <= LON_MAX):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Longitude {request.longitude}°E is out of bounds. Supported range is {LON_MIN}°E to {LON_MAX}°E."
        )

    # 2. Run Real Model Inference
    try:
        result = INFERENCE_SERVICE.predict(
            date_str=request.date,
            lat=request.latitude,
            lon=request.longitude
        )
    except KeyError as ke:
        # Date unavailable
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ke).strip("'\"")
        )
    except Exception as exc:
        logger.error(f"Unexpected inference failure for request {request}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference failure during subsurface reconstruction."
        )

    # 3. Handle Land Cells or Missing Ocean Data
    if not result.get("is_valid_ocean", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Target location is on land or has invalid satellite observations.")
        )

    return PredictionResponse(**result)


@app.get("/", summary="Root Documentation Link", tags=["System"])
def root_endpoint():
    """Welcome endpoint providing service name and documentation URL."""
    return {
        "service": "OceanEmbed Subsurface Temperature Reconstruction API",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "status": "operational",
        "documentation": "/docs",
        "openapi_spec": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
