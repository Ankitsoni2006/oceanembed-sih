"""
SIH26066 - OceanEmbed ARGO 2020 Observational Catalog
=====================================================

Reusable, in-process-cached index of the authentic September 2020 ARGO profiling
float observations that are used for INDEPENDENT OBSERVATIONAL EVALUATION of the
frozen OceanEmbed v3 subsurface temperature reconstruction model.

SCIENTIFIC CONTRACT
-------------------
* ARGO is an EVALUATION SOURCE ONLY. Nothing in this module is ever fed to the
  model as an input, and nothing here participates in training or model selection.
* Missing observations are preserved as missing (``None`` / ``NaN``). No value is
  ever fabricated outside the documented vertical interpolation onto the 15 SIH
  standard depths.
* This module performs no model inference and no metric computation. Those live in
  :mod:`backend.argo`, which consumes this catalog.

The vertical interpolation and quality filtering rules below deliberately mirror
``scripts/evaluate_argo2020_validation.py`` so that the interactive API reproduces
the same authentic evaluation set (36 profiles / 28 unique WMO floats).
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("OceanEmbed.ArgoCatalog")

# ---------------------------------------------------------------------------
# Domain constants (single source of truth = src.data.catalog)
# ---------------------------------------------------------------------------
from src.data.catalog import TARGET_DEPTHS, TARGET_GRID

#: The 15 SIH standard target depths, as float64.
TARGET_DEPTH_VALUES: np.ndarray = np.array(list(TARGET_DEPTHS.depths), dtype=np.float64)

#: Default location of the authentic contemporaneous September 2020 ARGO archive.
DEFAULT_ARGO_DIR: str = os.path.join("data", "argo", "argo2020")

# ---------------------------------------------------------------------------
# Authentic-observation quality rules (mirrors the existing offline pipeline)
# ---------------------------------------------------------------------------
#: Minimum physically valid pressure/temperature pairs for a genuine CTD cast.
MIN_VALID_LEVELS: int = 10

#: A North Indian Ocean profile whose warmest valid sample is colder than this is
#: treated as an uncalibrated/failed sensor transmission (the dummy 0.0 degC case).
MIN_PROFILE_MAX_TEMP_C: float = 15.0

#: Physically admissible temperature window for this tropical basin.
MIN_PHYSICAL_TEMP_C: float = 2.0
MAX_PHYSICAL_TEMP_C: float = 40.0

#: A profile must yield at least this many valid depths on the 15-level grid.
MIN_VALID_TARGET_DEPTHS: int = 5

#: Vertical extrapolation margins around the observed pressure envelope (dbar).
DEPTH_EXTRAPOLATION_MARGIN_DBAR: float = 10.0   # allow slightly below deepest sample
SURFACE_EXTRAPOLATION_MARGIN_DBAR: float = 5.0  # allow slightly above shallowest sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decode_char_variable(value: Any) -> str:
    """
    Decodes an ARGO NetCDF char variable element into a clean string.

    ARGO ``PLATFORM_NUMBER`` / ``*_QC`` fields are returned by xarray as an
    ``object`` array holding ``bytes``. Iterating a ``bytes`` object yields
    integers, so concatenating ``str(c)`` over its elements corrupts the value
    (a real defect present in the earlier offline scripts, which is why the
    older matching ledger records WMO IDs such as ``5057485049555232``). This
    helper decodes the whole element once, which is the correct behaviour.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        return decode_char_variable(bytes(np.asarray(value).tolist()))
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace").strip()
    return str(value).strip()


def is_inside_nio_domain(lat: float, lon: float) -> bool:
    """True when a coordinate falls inside the SIH North Indian Ocean domain."""
    return (
        TARGET_GRID.lat_min <= lat <= TARGET_GRID.lat_max
        and TARGET_GRID.lon_min <= lon <= TARGET_GRID.lon_max
    )


def interpolate_to_target_depths(
    pressure: np.ndarray,
    temperature: np.ndarray,
    target_depths: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Interpolates a float's (pressure, temperature) cast onto the 15 SIH standard
    depths, using the same guardrails as the existing offline ARGO evaluation.

    Missing target depths are returned as ``NaN`` - they are never filled.
    Pressure (dbar) is used directly as depth (m), the standard approximation for
    these upper-ocean ARGO casts.

    Returns
    -------
    np.ndarray of shape ``(15,)``, ``NaN`` where no observation is available.
    """
    target = (
        TARGET_DEPTH_VALUES if target_depths is None
        else np.asarray(target_depths, dtype=np.float64)
    )

    pressure = np.asarray(pressure, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)

    t_interp = np.full(target.shape, np.nan, dtype=np.float64)
    if pressure.size < 2:
        return t_interp

    order = np.argsort(pressure)
    p_sorted = pressure[order]
    t_sorted = temperature[order]
    if np.unique(p_sorted).size < 2:
        return t_interp

    p_min = float(p_sorted.min())
    p_max = float(p_sorted.max())

    t_interp = np.interp(target, p_sorted, t_sorted, left=np.nan, right=np.nan)

    # Enforce the observed sensor envelope: never extrapolate beyond it.
    for d_i, d_val in enumerate(target):
        if d_val > (p_max + DEPTH_EXTRAPOLATION_MARGIN_DBAR):
            t_interp[d_i] = np.nan
        if d_val < (p_min - SURFACE_EXTRAPOLATION_MARGIN_DBAR):
            t_interp[d_i] = np.nan

    return t_interp


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ArgoProfile:
    """A single authentic ARGO profiling-float cast inside the NIO domain."""

    profile_id: str
    wmo: str
    timestamp: str                 # ISO-8601 UTC "YYYY-MM-DDTHH:MM:SS"
    date: str                      # "YYYY-MM-DD"
    latitude: float
    longitude: float
    source_file: str
    cycle_number: Optional[int]
    data_mode: str
    valid_observation_count: int   # levels carrying a physical pressure/temperature pair
    valid_target_depth_count: int  # of the 15 SIH depths, how many are genuinely observed
    p_min_dbar: float
    p_max_dbar: float
    observed_temperatures: List[Optional[float]]       # length 15, None where missing
    observed_pressures: List[Optional[float]]          # raw cast pressures (valid pairs only)
    observed_temperatures_raw: List[Optional[float]]   # raw cast temperatures (valid pairs only)

    @property
    def observed_depths_m(self) -> List[int]:
        """Target depths that carry a genuine observation."""
        return [
            int(TARGET_DEPTH_VALUES[i])
            for i, t in enumerate(self.observed_temperatures)
            if t is not None
        ]

    def to_metadata(self) -> Dict[str, Any]:
        """Compact metadata payload for list endpoints (no numeric series)."""
        return {
            "profile_id": self.profile_id,
            "wmo": self.wmo,
            "timestamp": self.timestamp,
            "date": self.date,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source_file": self.source_file,
            "cycle_number": self.cycle_number,
            "data_mode": self.data_mode,
            "valid_observation_count": self.valid_observation_count,
            "valid_target_depth_count": self.valid_target_depth_count,
            "observed_depths_m": self.observed_depths_m,
            "p_min_dbar": self.p_min_dbar,
            "p_max_dbar": self.p_max_dbar,
        }


@dataclass
class ArgoCatalogDiagnostics:
    """Honest bookkeeping of everything the catalog read and rejected."""

    source_directory: str = ""
    source_files: List[str] = field(default_factory=list)
    total_raw_profiles_read: int = 0
    profiles_inside_nio_domain: int = 0
    profiles_outside_nio_domain: int = 0
    profiles_rejected_uncalibrated: int = 0
    profiles_rejected_insufficient_depths: int = 0
    rejected_wmo_ids: List[str] = field(default_factory=list)
    accepted_profiles: int = 0
    unique_wmo_count: int = 0
    observation_dates: List[str] = field(default_factory=list)
    min_timestamp: Optional[str] = None
    max_timestamp: Optional[str] = None
    total_valid_target_depth_observations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_iso_timestamp(value: Any) -> str:
    """Normalises a numpy datetime64 / datetime / string into ISO-8601 seconds."""
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return ""
        return np.datetime_as_string(value.astype("datetime64[s]"), unit="s")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    return str(value)[:19].replace(" ", "T")




# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
class ArgoCatalog:
    """
    Read-only index over the authentic ARGO NetCDF archive.

    ``build()`` is idempotent and is invoked at most once per process by the
    :func:`get_argo_catalog` singleton accessor, so NetCDF files are never
    re-parsed on every request.
    """

    def __init__(self, argo_dir: Optional[str] = None,
                 target_depths: Optional[Sequence[float]] = None):
        self.argo_dir = argo_dir or DEFAULT_ARGO_DIR
        self.target_depths = (
            TARGET_DEPTH_VALUES if target_depths is None
            else np.asarray(target_depths, dtype=np.float64)
        )
        self._profiles: Dict[str, ArgoProfile] = {}
        self._diagnostics = ArgoCatalogDiagnostics(source_directory=self.argo_dir)
        self._is_built = False

    # -- properties --------------------------------------------------------
    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def diagnostics(self) -> ArgoCatalogDiagnostics:
        return self._diagnostics

    @property
    def profiles(self) -> List[ArgoProfile]:
        return list(self._profiles.values())

    @property
    def profile_ids(self) -> List[str]:
        return list(self._profiles.keys())

    def __len__(self) -> int:
        return len(self._profiles)

    def get(self, profile_id: str) -> Optional[ArgoProfile]:
        return self._profiles.get(profile_id)

    # -- construction ------------------------------------------------------
    def build(self, force: bool = False) -> "ArgoCatalog":
        """Parses every ARGO NetCDF file in the archive exactly once."""
        if self._is_built and not force:
            return self

        import xarray as xr  # local import keeps module import cost low

        self._profiles = {}
        diag = ArgoCatalogDiagnostics(source_directory=self.argo_dir)

        file_paths = sorted(glob.glob(os.path.join(self.argo_dir, "*.nc")))
        diag.source_files = [os.path.basename(p) for p in file_paths]
        if not file_paths:
            raise FileNotFoundError(
                f"No ARGO NetCDF files found in '{self.argo_dir}'. Expected the "
                "contemporaneous September 2020 archive (20200901_prof.nc, "
                "20200915_prof.nc, 20200925_prof.nc)."
            )

        stamped: List[ArgoProfile] = []

        for file_path in file_paths:
            basename = os.path.basename(file_path)
            with xr.open_dataset(file_path) as ds:
                n_prof = int(ds["LATITUDE"].shape[0])
                diag.total_raw_profiles_read += n_prof

                lats = np.asarray(ds["LATITUDE"].values, dtype=np.float64)
                lons = np.asarray(ds["LONGITUDE"].values, dtype=np.float64)
                julds = np.asarray(ds["JULD"].values)
                platform = ds["PLATFORM_NUMBER"].values if "PLATFORM_NUMBER" in ds.variables else None
                cycles = ds["CYCLE_NUMBER"].values if "CYCLE_NUMBER" in ds.variables else None
                data_modes = ds["DATA_MODE"].values if "DATA_MODE" in ds.variables else None
                pres_all = np.asarray(ds["PRES"].values, dtype=np.float64)
                temp_all = np.asarray(ds["TEMP"].values, dtype=np.float64)

                for idx in range(n_prof):
                    lat = float(lats[idx])
                    lon = float(lons[idx])

                    if np.isnan(lat) or np.isnan(lon) or not is_inside_nio_domain(lat, lon):
                        diag.profiles_outside_nio_domain += 1
                        continue
                    diag.profiles_inside_nio_domain += 1

                    wmo = decode_char_variable(platform[idx]) if platform is not None else f"UNKNOWN{idx}"
                    if not wmo:
                        wmo = f"UNKNOWN{idx}"

                    cycle = (
                        int(cycles[idx])
                        if cycles is not None and not np.isnan(cycles[idx]) else None
                    )
                    data_mode = decode_char_variable(data_modes[idx]) if data_modes is not None else ""

                    pres = pres_all[idx]
                    temp = temp_all[idx]

                    valid = (
                        ~np.isnan(pres)
                        & ~np.isnan(temp)
                        & (pres >= 0)
                        & (temp >= MIN_PHYSICAL_TEMP_C)
                        & (temp < MAX_PHYSICAL_TEMP_C)
                    )
                    p_v = pres[valid]
                    t_v = temp[valid]

                    is_authentic = (
                        p_v.size >= MIN_VALID_LEVELS
                        and p_v.size > 0
                        and float(np.max(t_v)) >= MIN_PROFILE_MAX_TEMP_C
                    )
                    if not is_authentic:
                        diag.profiles_rejected_uncalibrated += 1
                        if wmo not in diag.rejected_wmo_ids:
                            diag.rejected_wmo_ids.append(wmo)
                        continue

                    t_interp = interpolate_to_target_depths(p_v, t_v, self.target_depths)
                    valid_target_count = int(np.sum(~np.isnan(t_interp)))
                    if valid_target_count < MIN_VALID_TARGET_DEPTHS:
                        diag.profiles_rejected_insufficient_depths += 1
                        if wmo not in diag.rejected_wmo_ids:
                            diag.rejected_wmo_ids.append(wmo)
                        continue

                    timestamp = _to_iso_timestamp(julds[idx])
                    stamped.append(
                        ArgoProfile(
                            profile_id=f"{basename}_{idx}",
                            wmo=wmo,
                            timestamp=timestamp,
                            date=timestamp[:10],
                            latitude=round(lat, 4),
                            longitude=round(lon, 4),
                            source_file=basename,
                            cycle_number=cycle,
                            data_mode=data_mode,
                            valid_observation_count=int(p_v.size),
                            valid_target_depth_count=valid_target_count,
                            p_min_dbar=round(float(np.min(p_v)), 2),
                            p_max_dbar=round(float(np.max(p_v)), 2),
                            observed_temperatures=[
                                (None if np.isnan(v) else round(float(v), 4)) for v in t_interp
                            ],
                            observed_pressures=[
                                (None if np.isnan(v) else round(float(v), 2)) for v in p_v
                            ],
                            observed_temperatures_raw=[
                                (None if np.isnan(v) else round(float(v), 4)) for v in t_v
                            ],
                        )
                    )

        # Deterministic ordering: chronological, then by profile id.
        stamped.sort(key=lambda p: (p.timestamp, p.profile_id))
        self._profiles = {p.profile_id: p for p in stamped}

        diag.accepted_profiles = len(self._profiles)
        diag.unique_wmo_count = len({p.wmo for p in self._profiles.values()})
        diag.observation_dates = sorted({p.date for p in self._profiles.values()})
        if stamped:
            diag.min_timestamp = stamped[0].timestamp
            diag.max_timestamp = stamped[-1].timestamp
        diag.total_valid_target_depth_observations = int(
            sum(p.valid_target_depth_count for p in self._profiles.values())
        )

        self._diagnostics = diag
        self._is_built = True

        logger.info(
            "ARGO catalog built: %d authentic profiles, %d unique WMO floats, "
            "%d valid profile-depth observations across dates %s",
            diag.accepted_profiles,
            diag.unique_wmo_count,
            diag.total_valid_target_depth_observations,
            diag.observation_dates,
        )
        return self

    # -- convenience -------------------------------------------------------
    def export_index_payload(self) -> Dict[str, Any]:
        """Serialisable snapshot of the catalog (for offline inspection)."""
        self.build()
        return {
            "source": "Coriolis / INCOIS Global Data Assembly Centre (GDAC)",
            "provenance_note": (
                "Authentic September 2020 ARGO profiling-float observations used strictly as an "
                "independent post-training observational check. ARGO is never a model input."
            ),
            "target_depths_m": [int(d) for d in self.target_depths],
            "diagnostics": self._diagnostics.to_dict(),
            "profiles": [p.to_metadata() for p in self.profiles],
        }


# ---------------------------------------------------------------------------
# Process-wide singleton (parse ARGO once, never per request)
# ---------------------------------------------------------------------------
_CATALOG_SINGLETON: Optional[ArgoCatalog] = None


def get_argo_catalog(argo_dir: Optional[str] = None) -> ArgoCatalog:
    """Returns the process-wide ARGO catalog, building it on first use."""
    global _CATALOG_SINGLETON
    if _CATALOG_SINGLETON is None:
        _CATALOG_SINGLETON = ArgoCatalog(argo_dir=argo_dir)
    if not _CATALOG_SINGLETON.is_built:
        _CATALOG_SINGLETON.build()
    return _CATALOG_SINGLETON


__all__ = [
    "ArgoCatalog",
    "ArgoCatalogDiagnostics",
    "ArgoProfile",
    "TARGET_DEPTH_VALUES",
    "DEFAULT_ARGO_DIR",
    "decode_char_variable",
    "get_argo_catalog",
    "interpolate_to_target_depths",
    "is_inside_nio_domain",
]
