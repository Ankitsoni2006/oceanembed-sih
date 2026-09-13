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
