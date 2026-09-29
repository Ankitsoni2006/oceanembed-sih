/**
 * SIH26066 — OceanEmbed TypeScript Data Contracts
 * Exactly maps to backend/schemas.py Pydantic models.
 */

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  model: string;
}

export interface RegionInfo {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

export interface ModelInfoResponse {
  model: string;
  version: string;
  parameters: number;
  input_variables: string[];
  output_depths_m: number[];
  grid_resolution: string;
  region: RegionInfo;
  device: string;
  checkpoint: string;
  status: string;
}

export interface AvailableDatesResponse {
  total_dates: number;
  date_range: {
    start: string;
    end: string;
  };
  dates: string[];
}

export interface PredictionRequest {
  date: string;
  latitude: number;
  longitude: number;
}

export interface SurfaceObservations {
  sst_c: number | null;
  sss_psu: number | null;
  ssh_m: number | null;
  u_current_ms: number | null;
  v_current_ms: number | null;
  u_wind_ms: number | null;
  v_wind_ms: number | null;
}

export interface OceanographicIndicators {
  mixed_layer_depth_m: number | null;
  thermocline_depth_m: number | null;
  ocean_heat_content_300m_gj_m2: number | null;
}

export interface PredictionResponse {
  date: string;
  latitude: number;
  longitude: number;
  grid_latitude: number;
  grid_longitude: number;
  depths_m: number[];
  temperatures_c: number[];
  model: string;
  model_version: string;
  inference_ms: number;
  total_latency_ms: number;
  is_valid_ocean: boolean;
  surface_observations?: SurfaceObservations | null;
  oceanographic_indicators?: OceanographicIndicators | null;
}

export interface DepthPoint {
  depth_m: number;
  temperature_c: number;
  isThermocline: boolean;
}

// ===========================================================================
// ARGO Independent Observational Evaluation — mirrors backend/schemas.py
// ARGO is an EVALUATION-ONLY source. It is never a model input.
// Every value below is returned at request time by the FastAPI backend.
// ===========================================================================

/** Metadata for one authentic ARGO profiling-float cast (ArgoProfileMetadata). */
export interface ArgoProfile {
  profile_id: string;
  wmo: string;
  timestamp: string;
  date: string;
  latitude: number;
  longitude: number;
  source_file: string;
  cycle_number: number | null;
  data_mode: string;
  valid_observation_count: number;
  valid_target_depth_count: number;
  observed_depths_m: number[];
  p_min_dbar: number;
  p_max_dbar: number;
}

/** GET /argo/profiles (ArgoProfileListResponse) */
export interface ArgoProfileListResponse {
  evaluation_type: string;
  source: string;
  provenance_note: string;
  target_depths_m: number[];
  total_profiles: number;
  unique_wmo_count: number;
  observation_dates: string[];
  total_valid_observations: number;
  source_files: string[];
  profiles: ArgoProfile[];
}

/** Which of the 7 surface satellite variables are present at the matched grid cell. */
export interface SurfaceInputAvailability {
  available_count: number;
  total_count: number;
  is_complete: boolean;
  available_channels: string[];
  missing_channels: string[];
}

/** Per-profile error statistics (ArgoMetrics). Nulls occur when undefined. */
export interface ArgoMetrics {
  count: number;
  rmse: number | null;
  mae: number | null;
  bias: number | null;
  corr: number | null;
}

/** One standard depth of a single-profile comparison (ArgoDepthComparisonRow). */
export interface DepthComparison {
  depth_m: number;
  observed_c: number | null;
  predicted_c: number | null;
  error_c: number | null;
  observed: boolean;
}

/** GET /argo/compare/{profile_id} (ArgoComparisonResponse) */
export interface ArgoProfileComparison {
  evaluation_type: string;
  source: string;
  model: string;
  model_version: string;
  argo_profile: ArgoProfile;
  matched_model_date: string;
  temporal_offset_hours: number;
  spatial_offset_km: number;
  grid_latitude: number;
  grid_longitude: number;
  grid_index: Record<string, number>;
  is_valid_ocean: boolean;
  surface_input_availability: SurfaceInputAvailability;
  depths_m: number[];
  depth_comparison: DepthComparison[];
  metrics: ArgoMetrics;
  surface_observations_at_grid_cell: SurfaceObservations | null;
  evaluation_ms: number;
}

/** Aggregate error statistics at one standard depth across the whole ARGO set. */
export interface ArgoDepthWiseMetric {
  depth_m: number;
  count: number;
  rmse: number | null;
  mae: number | null;
  bias: number | null;
  corr: number | null;
}

/** A profile whose matched grid cell is missing some surface satellite input. */
export interface ArgoDegradedInputProfile {
  profile_id: string;
  wmo: string;
  available_count: number;
  missing_channels: string[];
}

/** Bookkeeping and collocation statistics for the aggregate evaluation. */
export interface ArgoSummaryProfileMetadata {
  source_files: string[];
  observation_dates: string[];
  observation_window: Record<string, string | null>;
  raw_profiles_read: number;
  profiles_inside_nio_domain: number;
  profiles_rejected_uncalibrated: number;
  profiles_rejected_insufficient_depths: number;
  matched_model_dates: string[];
  max_temporal_offset_hours: number | null;
  mean_temporal_offset_hours: number | null;
  min_spatial_offset_km: number | null;
  mean_spatial_offset_km: number | null;
  max_spatial_offset_km: number | null;
  profiles_with_degraded_surface_inputs: ArgoDegradedInputProfile[];
  profiles_mapped_to_non_ocean_cells: string[];
  input_completeness_note: string;
}

/** A catalog profile excluded from the aggregate, with the reason. */
export interface ArgoSkippedProfile {
  profile_id: string;
  reason: string;
}

/** GET /argo/summary (ArgoSummaryResponse) */
export interface ArgoSummary {
  evaluation_type: string;
  source: string;
  model: string;
  model_version: string;
  note: string;
  profile_count: number;
  unique_wmo_count: number;
  matched_observation_count: number;
  rmse: number | null;
  mae: number | null;
  bias: number | null;
  pearson_r: number | null;
  depth_wise: ArgoDepthWiseMetric[];
  profile_metadata: ArgoSummaryProfileMetadata;
  skipped_profiles: ArgoSkippedProfile[];
  evaluation_ms: number;
}
