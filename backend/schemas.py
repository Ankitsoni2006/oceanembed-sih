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


# ===========================================================================
# ARGO Independent Observational Evaluation Schemas
# ---------------------------------------------------------------------------
# ARGO is an EVALUATION-ONLY source. It is never a model input. Every field
# below is produced at request time from authentic ARGO observations and a
# fresh forward pass of the frozen OceanEmbedNetV3_Decoder.
# ===========================================================================

class ArgoProfileMetadata(BaseModel):
    """Metadata for one authentic ARGO profiling-float cast."""

    profile_id: str = Field(..., description="Stable catalog identifier, e.g. '20200901_prof.nc_101'")
    wmo: str = Field(..., description="WMO platform number of the profiling float")
    timestamp: str = Field(..., description="Observation timestamp (UTC, ISO-8601)")
    date: str = Field(..., description="Observation calendar date (YYYY-MM-DD)")
    latitude: float = Field(..., description="Float latitude at profile time (degrees North)")
    longitude: float = Field(..., description="Float longitude at profile time (degrees East)")
    source_file: str = Field(..., description="Origin NetCDF file in data/argo/argo2020/")
    cycle_number: Optional[int] = Field(None, description="ARGO cycle number, when available")
    data_mode: str = Field("", description="ARGO data mode flag ('D' = delayed-mode quality controlled)")
    valid_observation_count: int = Field(..., description="Number of physically valid pressure/temperature samples")
    valid_target_depth_count: int = Field(..., description="Of the 15 standard depths, how many carry a genuine observation")
    observed_depths_m: List[int] = Field(..., description="Standard depths that carry a genuine observation")
    p_min_dbar: float = Field(..., description="Shallowest observed pressure (dbar)")
    p_max_dbar: float = Field(..., description="Deepest observed pressure (dbar)")


class ArgoProfileListResponse(BaseModel):
    """The complete authentic ARGO evaluation set available for interaction."""

    evaluation_type: str = Field(..., description="Always INDEPENDENT_OFFLINE_ARGO_OBSERVATIONAL_EVALUATION")
    source: str = Field(..., description="Data provenance statement")
    provenance_note: str = Field(..., description="Explicit statement of the offline, non-live nature of the ARGO set")
    target_depths_m: List[int] = Field(..., description="The 15 standard reconstruction depths")
    total_profiles: int = Field(..., description="Number of authentic profiles in the catalog")
    unique_wmo_count: int = Field(..., description="Number of distinct WMO profiling floats represented")
    observation_dates: List[str] = Field(..., description="Calendar days on which observations were collected")
    total_valid_observations: int = Field(..., description="Total authentic profile-depth observations available")
    source_files: List[str] = Field(..., description="NetCDF archive files indexed")
    profiles: List[ArgoProfileMetadata] = Field(..., description="Per-profile metadata, chronologically sorted")


class SurfaceInputAvailability(BaseModel):
    """Which of the 7 surface satellite variables are present at the matched grid cell."""

    available_count: int = Field(..., description="Number of surface variables present at this grid cell")
    total_count: int = Field(..., description="Total number of surface variables the model consumes")
    is_complete: bool = Field(..., description="True when all surface variables are present")
    available_channels: List[str] = Field(..., description="Surface variables present at this grid cell")
    missing_channels: List[str] = Field(..., description="Surface variables masked out at this grid cell")


class ArgoMetrics(BaseModel):
    """Paired observation-vs-prediction statistics (NaNs are excluded per pair)."""

    count: int = Field(..., description="Number of valid matched observation/prediction pairs")
    rmse: Optional[float] = Field(None, description="Root mean square error (degC)")
    mae: Optional[float] = Field(None, description="Mean absolute error (degC)")
    bias: Optional[float] = Field(None, description="Mean signed error, prediction minus observation (degC)")
    corr: Optional[float] = Field(None, description="Pearson correlation coefficient, when defined")


class ArgoDepthComparisonRow(BaseModel):
    """One standard depth of a single-profile comparison."""

    depth_m: int = Field(..., description="Target depth (meters)")
    observed_c: Optional[float] = Field(None, description="ARGO observed temperature (degC); null when not observed")
    predicted_c: Optional[float] = Field(None, description="OceanEmbed v3 reconstructed temperature (degC)")
    error_c: Optional[float] = Field(None, description="Prediction minus observation (degC); null when not observed")
    observed: bool = Field(..., description="Whether a genuine ARGO observation exists at this depth")


class ArgoComparisonResponse(BaseModel):
    """Full comparison of one authentic ARGO profile against the frozen model."""

    evaluation_type: str = Field(..., description="Always INDEPENDENT_OFFLINE_ARGO_OBSERVATIONAL_EVALUATION")
    source: str = Field(..., description="Data provenance statement")
    model: str = Field(..., description="Active model architecture")
    model_version: str = Field(..., description="Active model version")
    argo_profile: ArgoProfileMetadata = Field(..., description="Metadata for the compared ARGO profile")
    matched_model_date: str = Field(..., description="Processed surface-input date used for the reconstruction")
    temporal_offset_hours: float = Field(..., description="Absolute time difference from the 12:00 UTC daily-mean centre")
    spatial_offset_km: float = Field(..., description="Haversine distance from the float to the matched grid-cell centre")
    grid_latitude: float = Field(..., description="Matched canonical grid latitude")
    grid_longitude: float = Field(..., description="Matched canonical grid longitude")
    grid_index: Dict[str, int] = Field(..., description="Matched grid indices into the [15, 101, 241] reconstruction")
    is_valid_ocean: bool = Field(..., description="Whether the matched grid cell carries valid satellite observations")
    surface_input_availability: SurfaceInputAvailability = Field(..., description="Input-completeness diagnostic for the matched cell")
    depths_m: List[int] = Field(..., description="The 15 standard reconstruction depths")
    depth_comparison: List[ArgoDepthComparisonRow] = Field(..., description="Observed vs predicted at each standard depth")
    metrics: ArgoMetrics = Field(..., description="Dynamically computed error statistics for this profile")
    surface_observations_at_grid_cell: Optional[SurfaceObservations] = Field(None, description="Raw surface inputs the model consumed at this cell")
    evaluation_ms: float = Field(..., description="Server-side time to run this comparison (milliseconds)")


class ArgoDepthWiseMetric(BaseModel):
    """Aggregate error statistics at one standard depth across the whole ARGO set."""

    depth_m: int = Field(..., description="Target depth (meters)")
    count: int = Field(..., description="Matched observations contributing at this depth")
    rmse: Optional[float] = Field(None, description="Root mean square error (degC)")
    mae: Optional[float] = Field(None, description="Mean absolute error (degC)")
    bias: Optional[float] = Field(None, description="Mean signed error (degC)")
    corr: Optional[float] = Field(None, description="Pearson correlation coefficient, when defined")


class ArgoDegradedInputProfile(BaseModel):
    """A profile whose matched grid cell is missing some surface satellite input."""

    profile_id: str = Field(..., description="ARGO profile identifier")
    wmo: str = Field(..., description="WMO platform number")
    available_count: int = Field(..., description="Surface variables present at the matched cell")
    missing_channels: List[str] = Field(..., description="Surface variables absent at the matched cell")


class ArgoSummaryProfileMetadata(BaseModel):
    """Bookkeeping and collocation statistics for the aggregate evaluation."""

    source_files: List[str] = Field(..., description="ARGO NetCDF files contributing to the evaluation set")
    observation_dates: List[str] = Field(..., description="Calendar days on which observations were collected")
    observation_window: Dict[str, Optional[str]] = Field(..., description="Earliest and latest observation timestamps")
    raw_profiles_read: int = Field(..., description="Total profiles read from the archive across all basins")
    profiles_inside_nio_domain: int = Field(..., description="Profiles located inside the North Indian Ocean domain")
    profiles_rejected_uncalibrated: int = Field(..., description="Profiles rejected as uncalibrated or partial transmissions")
    profiles_rejected_insufficient_depths: int = Field(..., description="Profiles rejected for too few valid standard depths")
    matched_model_dates: List[str] = Field(..., description="Processed surface-input dates used for reconstruction")
    max_temporal_offset_hours: Optional[float] = Field(None, description="Largest observation-to-daily-mean time offset")
    mean_temporal_offset_hours: Optional[float] = Field(None, description="Mean observation-to-daily-mean time offset")
    min_spatial_offset_km: Optional[float] = Field(None, description="Smallest float-to-grid-centre distance")
    mean_spatial_offset_km: Optional[float] = Field(None, description="Mean float-to-grid-centre distance")
    max_spatial_offset_km: Optional[float] = Field(None, description="Largest float-to-grid-centre distance")
    profiles_with_degraded_surface_inputs: List[ArgoDegradedInputProfile] = Field(..., description="Profiles whose matched cell is missing some satellite input")
    profiles_mapped_to_non_ocean_cells: List[str] = Field(..., description="Profiles whose matched cell carries no valid SST mask")
    input_completeness_note: str = Field(..., description="Explanation of how partially masked input cells are handled")


class ArgoSkippedProfile(BaseModel):
    """A catalog profile that could not be evaluated, with the reason."""

    profile_id: str = Field(..., description="ARGO profile identifier")
    reason: str = Field(..., description="Why the profile was excluded from the aggregate")


class ArgoSummaryResponse(BaseModel):
    """Aggregate independent ARGO observational evaluation of the frozen model."""

    evaluation_type: str = Field(..., description="Always INDEPENDENT_OFFLINE_ARGO_OBSERVATIONAL_EVALUATION")
    source: str = Field(..., description="Data provenance statement")
    model: str = Field(..., description="Active model architecture")
    model_version: str = Field(..., description="Active model version")
    note: str = Field(..., description="Statement that statistics are computed at request time, not hardcoded")
    profile_count: int = Field(..., description="Authentic ARGO profiles evaluated")
    unique_wmo_count: int = Field(..., description="Distinct WMO profiling floats represented")
    matched_observation_count: int = Field(..., description="Matched profile-depth observations contributing to the statistics")
    rmse: Optional[float] = Field(None, description="Aggregate root mean square error (degC)")
    mae: Optional[float] = Field(None, description="Aggregate mean absolute error (degC)")
    bias: Optional[float] = Field(None, description="Aggregate mean signed error (degC)")
    pearson_r: Optional[float] = Field(None, description="Aggregate Pearson correlation coefficient")
    depth_wise: List[ArgoDepthWiseMetric] = Field(..., description="Aggregate error statistics at each standard depth")
    profile_metadata: ArgoSummaryProfileMetadata = Field(..., description="Bookkeeping and collocation statistics")
    skipped_profiles: List[ArgoSkippedProfile] = Field(..., description="Profiles excluded from the aggregate")
    evaluation_ms: float = Field(..., description="Server-side time to compute the aggregate (milliseconds)")



