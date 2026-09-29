"""
SIH26066 - OceanEmbed Interactive ARGO Observational Evaluation Service
=======================================================================

Serves the independent in-situ ARGO observational evaluation of the FROZEN
OceanEmbed v3 subsurface temperature reconstruction model.

SCIENTIFIC CONTRACT
-------------------
* ARGO observations are used ONLY as an independent post-training evaluation
  reference. They are never model inputs, never used for training, and never
  used for model or checkpoint selection.
* Every reported statistic is computed at request time from the authentic ARGO
  observations and a fresh forward pass of the frozen model. No benchmark value
  is hardcoded anywhere in this module.
* The served model, scaler, checkpoint and chunk LRU cache are the exact same
  singletons used by ``POST /predict``. The model is loaded once and never
  reloaded per ARGO request.
* ARGO NetCDF files are parsed once per process by the shared catalog and cached.
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from backend.config import TARGET_DEPTHS, INPUT_VARIABLES
from backend.inference import INFERENCE_SERVICE
from src.evaluation.metrics import compute_paired_metrics
from src.validation.argo_catalog import ArgoCatalog, ArgoProfile, get_argo_catalog

logger = logging.getLogger("OceanEmbed.ArgoValidation")

#: Mask channel offset: channels 7..13 are the validity masks for channels 0..6.
_MASK_CHANNEL_OFFSET = 7

#: Minimum valid paired observations required before a per-depth RMSE is reported.
MIN_PAIRED_OBSERVATIONS: int = 3

#: Human-readable provenance string surfaced by the API.
ARGO_SOURCE_NOTE: str = (
    "Coriolis / INCOIS Global Data Assembly Centre (GDAC) - September 2020 "
    "North Indian Ocean profiling floats. Offline observational evaluation set; "
    "this prototype does not provide a live or global ARGO feed."
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates in kilometres."""
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_km * c


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


class ArgoValidationService:
    """
    Singleton service that evaluates the frozen OceanEmbed v3 model against the
    authentic September 2020 ARGO profiling-float archive.

    Performance model
    -----------------
    ARGO profiles cluster on only three calendar days, so a complete aggregate
    evaluation requires just one full-grid forward pass per distinct date. Each
    reconstructed ``[15, 101, 241]`` field is LRU-cached, making repeated
    ``/argo/compare`` calls and the ``/argo/summary`` aggregate effectively free
    after warm-up.
    """

    def __init__(self) -> None:
        self._catalog: Optional[ArgoCatalog] = None
        self._prediction_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.max_cached_prediction_grids: int = 4
        self._summary_cache: Optional[Dict[str, Any]] = None
        self._last_summary_ms: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def catalog(self) -> ArgoCatalog:
        """Lazily built, process-wide ARGO catalog (NetCDF parsed once)."""
        if self._catalog is None:
            self._catalog = get_argo_catalog()
        return self._catalog

    @property
    def is_ready(self) -> bool:
        return self._catalog is not None and self._catalog.is_built

    def initialize(self) -> None:
        """Eagerly builds the ARGO catalog and warms up the model singleton."""
        if not INFERENCE_SERVICE.is_initialized:
            INFERENCE_SERVICE.initialize()
        self.catalog.build()
        logger.info(
            "ArgoValidationService ready: %d authentic profiles, %d unique WMO floats.",
            self.catalog.diagnostics.accepted_profiles,
            self.catalog.diagnostics.unique_wmo_count,
        )

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------
    def list_profiles(self) -> List[Dict[str, Any]]:
        """Metadata for every authentic ARGO profile in the evaluation set."""
        return [p.to_metadata() for p in self.catalog.profiles]

    def get_profile(self, profile_id: str) -> Optional[ArgoProfile]:
        return self.catalog.get(profile_id)

    def diagnostics(self) -> Dict[str, Any]:
        return self.catalog.diagnostics.to_dict()

    # ------------------------------------------------------------------
    # Model inference (reuses the /predict singleton; never reloads the model)
    # ------------------------------------------------------------------
    def _predict_grid_for_date(self, date_str: str) -> np.ndarray:
        """
        Returns the frozen model's full ``[15, 101, 241]`` reconstruction for a
        date, from the LRU cache when available.

        The forward pass is identical to the one ``/predict`` performs; the full
        field is computed once and reused for every profile matched to that date.
        """
        cached = self._prediction_cache.get(date_str)
        if cached is not None:
            self._prediction_cache.move_to_end(date_str)
            return cached

        if not INFERENCE_SERVICE.is_initialized:
            INFERENCE_SERVICE.initialize()

        raw_day = INFERENCE_SERVICE.get_raw_day_tensor(date_str)       # [14, 101, 241] CPU
        scaler = INFERENCE_SERVICE.get_fitted_scaler()
        scaled = scaler.transform(raw_day.unsqueeze(0)).to(INFERENCE_SERVICE.device)

        with torch.inference_mode():
            prediction = INFERENCE_SERVICE.model(scaled)                # [1, 15, 101, 241]

        grid = prediction[0].detach().to("cpu").numpy().astype(np.float64)

        self._prediction_cache[date_str] = grid
        if len(self._prediction_cache) > self.max_cached_prediction_grids:
            self._prediction_cache.popitem(last=False)
        return grid

    def _surface_input_availability(
        self, raw_day: torch.Tensor, lat_idx: int, lon_idx: int
    ) -> Dict[str, Any]:
        """
        Reports which of the 7 surface satellite variables are actually present at
        a grid cell, using the input tensor's validity-mask channels.

        This is a transparency diagnostic, not a filter: an ARGO profile is never
        silently dropped because its host grid cell has missing satellite input.
        Cells with missing inputs are surfaced explicitly so a reader can judge how
        much of the error is attributable to degraded model input.
        """
        present = {}
        for ch, name in enumerate(INPUT_VARIABLES):
            mask_value = float(raw_day[ch + _MASK_CHANNEL_OFFSET, lat_idx, lon_idx].item())
            present[name] = bool(mask_value > 0.5)

        available = sum(1 for v in present.values() if v)
        return {
            "available_count": available,
            "total_count": len(INPUT_VARIABLES),
            "is_complete": available == len(INPUT_VARIABLES),
            "available_channels": [name for name, ok in present.items() if ok],
            "missing_channels": [name for name, ok in present.items() if not ok],
        }

    def _matched_prediction_column(
        self,
        profile: ArgoProfile,
    ) -> Tuple[str, np.ndarray, int, int, float, float, float, Dict[str, float], bool, Dict[str, Any]]:
        """
        Runs the frozen model for the profile's observation day and extracts the
        15-depth column at the profile's nearest 0.25 deg grid cell.

        Returns
        -------
        (matched_date, prediction_column, lat_idx, lon_idx, grid_lat, grid_lon,
         spatial_offset_km, surface_observations, is_valid_ocean, input_availability)
        """
        matched_date, _ = INFERENCE_SERVICE.resolve_nearest_available_date(profile.date)

        # Temporal offset against the 12:00 UTC centre of the daily-mean product.
        from datetime import datetime

        obs_dt = datetime.strptime(profile.timestamp, "%Y-%m-%dT%H:%M:%S")
        centre_dt = datetime.strptime(matched_date, "%Y-%m-%d").replace(hour=12, minute=0, second=0)
        temporal_offset_hours = abs((obs_dt - centre_dt).total_seconds()) / 3600.0

        # Spatial mapping to the canonical reconstruction grid.
        lat_idx, lon_idx, grid_lat, grid_lon = INFERENCE_SERVICE.map_coordinates_to_grid(
            profile.latitude, profile.longitude
        )
        spatial_offset_km = haversine_km(
            profile.latitude, profile.longitude, grid_lat, grid_lon
        )

        raw_day = INFERENCE_SERVICE.get_raw_day_tensor(matched_date)     # [14, 101, 241]
        is_valid_ocean = bool(float(raw_day[7, lat_idx, lon_idx].item()) > 0.5)
        input_availability = self._surface_input_availability(raw_day, lat_idx, lon_idx)

        surface_observations = {
            "sst_c": _round_or_none(raw_day[0, lat_idx, lon_idx].item(), 3),
            "sss_psu": _round_or_none(raw_day[1, lat_idx, lon_idx].item(), 3),
            "ssh_m": _round_or_none(raw_day[2, lat_idx, lon_idx].item(), 3),
            "u_current_ms": _round_or_none(raw_day[3, lat_idx, lon_idx].item(), 3),
            "v_current_ms": _round_or_none(raw_day[4, lat_idx, lon_idx].item(), 3),
            "u_wind_ms": _round_or_none(raw_day[5, lat_idx, lon_idx].item(), 3),
            "v_wind_ms": _round_or_none(raw_day[6, lat_idx, lon_idx].item(), 3),
        }

        grid = self._predict_grid_for_date(matched_date)
        column = grid[:, lat_idx, lon_idx]

        return (
            matched_date,
            column,
            lat_idx,
            lon_idx,
            grid_lat,
            grid_lon,
            spatial_offset_km,
            surface_observations,
            is_valid_ocean,
            input_availability,
        )


    # ------------------------------------------------------------------
    # Single-profile comparison
    # ------------------------------------------------------------------
    def compare(self, profile_id: str) -> Dict[str, Any]:
        """
        Compares one authentic ARGO profile against a fresh OceanEmbed v3
        reconstruction at the matched date and grid cell.

        Raises
        ------
        KeyError
            When ``profile_id`` is not present in the authentic ARGO catalog.
        """
        profile = self.catalog.get(profile_id)
        if profile is None:
            raise KeyError(
                f"ARGO profile '{profile_id}' is not present in the authentic "
                f"September 2020 evaluation catalog ({len(self.catalog)} profiles available)."
            )

        t_start = time.perf_counter()
        (
            matched_date,
            column,
            lat_idx,
            lon_idx,
            grid_lat,
            grid_lon,
            spatial_offset_km,
            surface_observations,
            is_valid_ocean,
            input_availability,
        ) = self._matched_prediction_column(profile)

        observed = np.array(
            [np.nan if v is None else float(v) for v in profile.observed_temperatures],
            dtype=np.float64,
        )
        predicted = np.array(column, dtype=np.float64)

        paired = ~np.isnan(observed)
        metrics = compute_paired_metrics(
            predicted[paired], observed[paired], min_samples=MIN_PAIRED_OBSERVATIONS
        )

        depth_rows: List[Dict[str, Any]] = []
        for d_i, depth_m in enumerate(TARGET_DEPTHS):
            obs_val = observed[d_i]
            pred_val = float(predicted[d_i])
            has_obs = not np.isnan(obs_val)
            depth_rows.append(
                {
                    "depth_m": int(depth_m),
                    "observed_c": _round_or_none(obs_val if has_obs else None, 4),
                    "predicted_c": _round_or_none(pred_val, 4),
                    "error_c": _round_or_none(pred_val - obs_val, 4) if has_obs else None,
                    "observed": bool(has_obs),
                }
            )

        from datetime import datetime

        obs_dt = datetime.strptime(profile.timestamp, "%Y-%m-%dT%H:%M:%S")
        centre_dt = datetime.strptime(matched_date, "%Y-%m-%d").replace(hour=12, minute=0, second=0)
        temporal_offset_hours = abs((obs_dt - centre_dt).total_seconds()) / 3600.0

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "evaluation_type": "INDEPENDENT_OFFLINE_ARGO_OBSERVATIONAL_EVALUATION",
            "source": ARGO_SOURCE_NOTE,
            "model": "OceanEmbedNetV3_Decoder",
            "model_version": "v3",
            "argo_profile": profile.to_metadata(),
            "matched_model_date": matched_date,
            "temporal_offset_hours": round(temporal_offset_hours, 2),
            "spatial_offset_km": round(spatial_offset_km, 2),
            "grid_latitude": grid_lat,
            "grid_longitude": grid_lon,
            "grid_index": {"lat": lat_idx, "lon": lon_idx},
            "is_valid_ocean": is_valid_ocean,
            "surface_input_availability": input_availability,
            "depths_m": list(TARGET_DEPTHS),
            "depth_comparison": depth_rows,
            "metrics": metrics,
            "surface_observations_at_grid_cell": surface_observations,
            "evaluation_ms": round(total_ms, 2),
        }

    # ------------------------------------------------------------------
    # Aggregate evaluation over the complete authentic ARGO set
    # ------------------------------------------------------------------
    def summary(self, force: bool = False) -> Dict[str, Any]:
        """
        Evaluates every authentic ARGO profile in the catalog against the frozen
        OceanEmbed v3 model and returns the aggregate statistics.

        The result is cached after the first computation so repeated UI loads do
        not re-run the evaluation. ``force=True`` recomputes from scratch.
        """
        if self._summary_cache is not None and not force:
            return self._summary_cache

        t_start = time.perf_counter()

        all_observed: List[float] = []
        all_predicted: List[float] = []
        per_depth_observed: Dict[int, List[float]] = {int(d): [] for d in TARGET_DEPTHS}
        per_depth_predicted: Dict[int, List[float]] = {int(d): [] for d in TARGET_DEPTHS}

        temporal_offsets: List[float] = []
        spatial_offsets: List[float] = []
        matched_dates: List[str] = []
        compared_profiles = 0
        skipped_profiles: List[Dict[str, str]] = []
        land_cell_profiles: List[str] = []
        incomplete_input_profiles: List[Dict[str, Any]] = []

        for profile in self.catalog.profiles:
            try:
                (
                    matched_date,
                    column,
                    _lat_idx,
                    _lon_idx,
                    _grid_lat,
                    _grid_lon,
                    spatial_offset_km,
                    _surface,
                    is_valid_ocean,
                    input_availability,
                ) = self._matched_prediction_column(profile)
            except KeyError as exc:
                skipped_profiles.append({"profile_id": profile.profile_id, "reason": str(exc)})
                continue

            compared_profiles += 1
            if not is_valid_ocean:
                land_cell_profiles.append(profile.profile_id)
            if not input_availability["is_complete"]:
                incomplete_input_profiles.append(
                    {
                        "profile_id": profile.profile_id,
                        "wmo": profile.wmo,
                        "available_count": input_availability["available_count"],
                        "missing_channels": input_availability["missing_channels"],
                    }
                )

            from datetime import datetime

            obs_dt = datetime.strptime(profile.timestamp, "%Y-%m-%dT%H:%M:%S")
            centre_dt = datetime.strptime(matched_date, "%Y-%m-%d").replace(
                hour=12, minute=0, second=0
            )
            temporal_offsets.append(abs((obs_dt - centre_dt).total_seconds()) / 3600.0)
            spatial_offsets.append(spatial_offset_km)
            if matched_date not in matched_dates:
                matched_dates.append(matched_date)

            for d_i, depth_m in enumerate(TARGET_DEPTHS):
                observed_val = profile.observed_temperatures[d_i]
                if observed_val is None:
                    continue
                predicted_val = float(column[d_i])
                if math.isnan(predicted_val) or math.isinf(predicted_val):
                    continue
                all_observed.append(float(observed_val))
                all_predicted.append(predicted_val)
                per_depth_observed[int(depth_m)].append(float(observed_val))
                per_depth_predicted[int(depth_m)].append(predicted_val)

        overall = compute_paired_metrics(all_predicted, all_observed, min_samples=MIN_PAIRED_OBSERVATIONS)

        depth_wise: List[Dict[str, Any]] = []
        for depth_m in TARGET_DEPTHS:
            d_int = int(depth_m)
            d_metrics = compute_paired_metrics(
                per_depth_predicted[d_int], per_depth_observed[d_int],
                min_samples=MIN_PAIRED_OBSERVATIONS,
            )
            depth_wise.append(
                {
                    "depth_m": d_int,
                    "count": d_metrics["count"],
                    "rmse": _round_or_none(d_metrics["rmse"]),
                    "mae": _round_or_none(d_metrics["mae"]),
                    "bias": _round_or_none(d_metrics["bias"]),
                    "corr": _round_or_none(d_metrics["corr"]),
                }
            )

        diag = self.catalog.diagnostics
        total_ms = (time.perf_counter() - t_start) * 1000.0

        payload = {
            "evaluation_type": "INDEPENDENT_OFFLINE_ARGO_OBSERVATIONAL_EVALUATION",
            "source": ARGO_SOURCE_NOTE,
            "model": "OceanEmbedNetV3_Decoder",
            "model_version": "v3",
            "note": (
                "Aggregate statistics are computed at request time from authentic ARGO observations "
                "and fresh forward passes of the frozen model. No benchmark value is hardcoded."
            ),
            "profile_count": compared_profiles,
            "unique_wmo_count": len({p.wmo for p in self.catalog.profiles}),
            "matched_observation_count": overall["count"],
            "rmse": _round_or_none(overall["rmse"]),
            "mae": _round_or_none(overall["mae"]),
            "bias": _round_or_none(overall["bias"]),
            "pearson_r": _round_or_none(overall["corr"]),
            "depth_wise": depth_wise,
            "profile_metadata": {
                "source_files": diag.source_files,
                "observation_dates": diag.observation_dates,
                "observation_window": {"start": diag.min_timestamp, "end": diag.max_timestamp},
                "raw_profiles_read": diag.total_raw_profiles_read,
                "profiles_inside_nio_domain": diag.profiles_inside_nio_domain,
                "profiles_rejected_uncalibrated": diag.profiles_rejected_uncalibrated,
                "profiles_rejected_insufficient_depths": diag.profiles_rejected_insufficient_depths,
                "matched_model_dates": sorted(matched_dates),
                "max_temporal_offset_hours": _round_or_none(
                    max(temporal_offsets) if temporal_offsets else None, 2
                ),
                "mean_temporal_offset_hours": _round_or_none(
                    float(np.mean(temporal_offsets)) if temporal_offsets else None, 2
                ),
                "min_spatial_offset_km": _round_or_none(
                    min(spatial_offsets) if spatial_offsets else None, 2
                ),
                "mean_spatial_offset_km": _round_or_none(
                    float(np.mean(spatial_offsets)) if spatial_offsets else None, 2
                ),
                "max_spatial_offset_km": _round_or_none(
                    max(spatial_offsets) if spatial_offsets else None, 2
                ),
                "profiles_with_degraded_surface_inputs": incomplete_input_profiles,
                "profiles_mapped_to_non_ocean_cells": land_cell_profiles,
                "input_completeness_note": (
                    "An ARGO profile is never dropped because its host grid cell has missing satellite "
                    "input. Cells whose input channels are partly masked are reported here so the "
                    "contribution of degraded model input to the aggregate error can be judged."
                ),
            },
            "skipped_profiles": skipped_profiles,
            "evaluation_ms": round(total_ms, 2),
        }

        self._summary_cache = payload
        self._last_summary_ms = total_ms
        logger.info(
            "ARGO aggregate evaluation complete: %d profiles, %d observations, RMSE %s degC (%.1f ms).",
            payload["profile_count"],
            payload["matched_observation_count"],
            payload["rmse"],
            total_ms,
        )
        return payload


# Global service instance (mirrors INFERENCE_SERVICE)
ARGO_VALIDATION_SERVICE = ArgoValidationService()

