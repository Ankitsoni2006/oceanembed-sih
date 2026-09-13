"""
SIH26066 — OceanEmbed Pydantic Schemas & DTOs
Defines strict request/response data contracts for all API endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service health status", examples=["ok"])
    model_loaded: bool = Field(True, description="Whether the ML model is loaded in memory")
    device: str = Field(..., description="Compute device currently hosting the model", examples=["cpu", "cuda:0"])
    model: str = Field("OceanEmbedNetV3_Decoder", description="Active model architecture name")


class RegionInfo(BaseModel):
    lat_min: float = Field(5.0, description="Minimum latitude (degrees North)")
    lat_max: float = Field(30.0, description="Maximum latitude (degrees North)")
    lon_min: float = Field(45.0, description="Minimum longitude (degrees East)")
    lon_max: float = Field(105.0, description="Maximum longitude (degrees East)")


class ModelInfoResponse(BaseModel):
    model: str = Field("OceanEmbedNetV3_Decoder", description="Model architecture identifier")
    version: str = Field("v3", description="Validated model version")
    parameters: int = Field(1275934, description="Total trainable parameter count")
    input_variables: List[str] = Field(..., description="7 surface input variables")
    output_depths_m: List[int] = Field(..., description="15 standard reconstruction depth levels (meters)")
    grid_resolution: str = Field("0.25°", description="Spatial resolution of the target reconstruction grid")
    region: RegionInfo = Field(..., description="Bounding box of the North Indian Ocean domain")
    device: str = Field(..., description="Device hosting model weights")
    checkpoint: str = Field(..., description="Relative path to model checkpoint")
    status: str = Field("validated", description="Validation and audit readiness status")


class DateRange(BaseModel):
    start: str = Field(..., description="Earliest available date in YYYY-MM-DD format", examples=["2020-01-01"])
    end: str = Field(..., description="Latest available date in YYYY-MM-DD format", examples=["2020-09-30"])


class AvailableDatesResponse(BaseModel):
    total_dates: int = Field(..., description="Number of daily ocean fields available for inference")
    date_range: DateRange = Field(..., description="Temporal coverage range")
    dates: List[str] = Field(..., description="Chronologically sorted list of supported date strings")


class PredictionRequest(BaseModel):
    date: str = Field(
        ...,
        description="Target observation date in YYYY-MM-DD format",
        examples=["2020-09-15"]
    )
    latitude: float = Field(
        ...,
        ge=5.0,
        le=30.0,
        description="Latitude in degrees North (5.0°N to 30.0°N)",
        examples=[15.0]
    )
    longitude: float = Field(
        ...,
        ge=45.0,
        le=105.0,
        description="Longitude in degrees East (45.0°E to 105.0°E)",
        examples=[85.0]
    )


class SurfaceObservations(BaseModel):
    sst_c: Optional[float] = Field(None, description="Sea Surface Temperature (°C)")
    sss_psu: Optional[float] = Field(None, description="Sea Surface Salinity (PSU)")
    ssh_m: Optional[float] = Field(None, description="Sea Surface Height Anomaly (m)")
    u_current_ms: Optional[float] = Field(None, description="Eastward surface current velocity (m/s)")
    v_current_ms: Optional[float] = Field(None, description="Northward surface current velocity (m/s)")
    u_wind_ms: Optional[float] = Field(None, description="10m eastward wind velocity (m/s)")
    v_wind_ms: Optional[float] = Field(None, description="10m northward wind velocity (m/s)")


class OceanographicIndicators(BaseModel):
    mixed_layer_depth_m: Optional[float] = Field(None, description="Estimated Mixed Layer Depth (meters) via 0.2°C temperature threshold")
    thermocline_depth_m: Optional[float] = Field(None, description="Depth of maximum vertical temperature gradient (meters)")
    ocean_heat_content_300m_gj_m2: Optional[float] = Field(None, description="Upper ocean heat content integrated from 0m to 300m (GJ/m²)")


class PredictionResponse(BaseModel):
    date: str = Field(..., description="Target date of reconstructed field", examples=["2020-09-15"])
    latitude: float = Field(..., description="Requested query latitude", examples=[15.0])
    longitude: float = Field(..., description="Requested query longitude", examples=[85.0])
    grid_latitude: float = Field(..., description="Mapped canonical 0.25° grid latitude", examples=[15.0])
    grid_longitude: float = Field(..., description="Mapped canonical 0.25° grid longitude", examples=[85.0])
    depths_m: List[int] = Field(..., description="15 vertical depth levels (meters)")
    temperatures_c: List[float] = Field(..., description="Reconstructed temperatures (°C) at the 15 standard depths")
    model: str = Field("OceanEmbedNetV3_Decoder", description="Model architecture used for inference")
    model_version: str = Field("v3", description="Model version")
    inference_ms: float = Field(..., description="Pure neural network forward inference execution time (milliseconds)")
    total_latency_ms: float = Field(..., description="Total endpoint execution time including preprocessing and extraction (milliseconds)")
    is_valid_ocean: bool = Field(True, description="Whether the requested location is a valid ocean water cell")
    surface_observations: Optional[SurfaceObservations] = Field(None, description="Surface satellite observations at this grid cell")
    oceanographic_indicators: Optional[OceanographicIndicators] = Field(None, description="Derived physical oceanographic indicators")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Short error code or description")
    message: str = Field(..., description="Human-readable explanation of why the request failed")
    detail: Optional[Any] = Field(None, description="Detailed diagnostic or coordinate context")
