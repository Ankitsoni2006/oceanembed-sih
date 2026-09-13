"""
SIH26066 — OceanEmbed Data Catalog
Authoritative specification of all input surface datasets, target GLORYS reanalysis,
and in-situ ARGO validation sources.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class TargetGridSpec:
    """Mathematical specification of the North Indian Ocean target grid."""
    lat_min: float = 5.0
    lat_max: float = 30.0
    lon_min: float = 45.0
    lon_max: float = 105.0
    resolution: float = 0.25
    
    @property
    def lats(self) -> np.ndarray:
        return np.arange(self.lat_min, self.lat_max + self.resolution / 2.0, self.resolution)
        
    @property
    def lons(self) -> np.ndarray:
        return np.arange(self.lon_min, self.lon_max + self.resolution / 2.0, self.resolution)
        
    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.lats), len(self.lons)


@dataclass(frozen=True)
class VerticalGridSpec:
    """The 15 locked SIH26066 target subsurface depths in meters."""
    depths: Tuple[float, ...] = (
        0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0,
        125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0
    )
    
    @property
    def num_depths(self) -> int:
        return len(self.depths)


@dataclass
class DatasetSpec:
    """Specification of an individual Copernicus or in-situ dataset."""
    name: str
    product_id: str
    dataset_id: str
    variables: List[str]
    units: Dict[str, str]
    native_spatial_res_deg: float
    temporal_frequency: str
    depth_range: Optional[Tuple[float, float]] = None
    depth_level_index: Optional[int] = None
    temporal_coverage: Tuple[str, str] = ("1993-01-01", "2025-12-31")
    description: str = ""


# Default Target Grid and Depths
TARGET_GRID = TargetGridSpec()
TARGET_DEPTHS = VerticalGridSpec()

# Common Verified Temporal Overlap
COMMON_USABLE_PERIOD = ("2010-06-03", "2025-12-25")

# Authoritative Catalog Entries
DATA_CATALOG: Dict[str, DatasetSpec] = {
    "SST": DatasetSpec(
        name="Sea Surface Temperature (OSTIA L4)",
        product_id="SST_GLO_SST_L4_REP_OBSERVATIONS_010_011",
        dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        variables=["analysed_sst"],
        units={"analysed_sst": "kelvin"},
        native_spatial_res_deg=0.05,
        temporal_frequency="daily",
        temporal_coverage=("1981-08-24", "2026-03-31"),
        description="Reprocessed OSTIA foundation SST merging satellite IR/MW and in-situ buoys"
    ),
    "SSS": DatasetSpec(
        name="Sea Surface Salinity (Multi-Obs L4)",
        product_id="MULTIOBS_GLO_PHY_REP_015_004",
        dataset_id="cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c",
        variables=["sss"],
        units={"sss": "psu"},
        native_spatial_res_deg=0.25,
        temporal_frequency="weekly",
        temporal_coverage=("2010-06-03", "2025-12-25"),
        description="Weekly objective analysis of SMOS and SMAP satellite radiometers"
    ),
    "SSH": DatasetSpec(
        name="Sea Surface Height / SLA (DUACS L4)",
        product_id="SEALEVEL_GLO_PHY_L4_MY_008_047",
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        variables=["sla"],
        units={"sla": "m"},
        native_spatial_res_deg=0.125,
        temporal_frequency="daily",
        temporal_coverage=("1993-01-01", "2026-03-31"),
        description="All-satellite merged gridded sea level anomaly"
    ),
    "CURRENTS": DatasetSpec(
        name="Surface Currents (Multi-Obs L4)",
        product_id="MULTIOBS_GLO_PHY_REP_015_004",
        dataset_id="cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m",
        variables=["uo", "vo"],
        units={"uo": "m/s", "vo": "m/s"},
        native_spatial_res_deg=0.25,
        temporal_frequency="daily",
        depth_level_index=0,  # 0m surface layer
        temporal_coverage=("1993-01-01", "2026-03-31"),
        description="Global total surface ocean currents combining altimetry and wind drifters"
    ),
    "WINDS": DatasetSpec(
        name="Surface Wind Stress (Scatterometer L4)",
        product_id="WIND_GLO_PHY_L4_MY_012_006",
        dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        variables=["eastward_wind", "northward_wind"],
        units={"eastward_wind": "m/s", "northward_wind": "m/s"},
        native_spatial_res_deg=0.125,
        temporal_frequency="hourly",
        temporal_coverage=("2007-01-11", "2026-04-21"),
        description="Gridded L4 hourly scatterometer winds aggregated to daily means"
    ),
    "GLORYS": DatasetSpec(
        name="GLORYS12V1 Global Reanalysis (3D Ground Truth Reference)",
        product_id="GLOBAL_MULTIYEAR_PHY_001_030",
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao"],
        units={"thetao": "degrees_C"},
        native_spatial_res_deg=0.083,
        temporal_frequency="daily",
        depth_range=(0.494, 1062.5),
        temporal_coverage=("1993-01-01", "2026-06-23"),
        description="Mercator ocean physical reanalysis 3D potential temperature"
    ),
}

# Input channels mapping (0 to 6 physical, 7 to 13 validity masks)
SURFACE_CHANNELS = [
    ("sst", "SST (OSTIA)", "°C"),
    ("sss", "SSS (Multi-Obs)", "psu"),
    ("ssh", "SSH/SLA (DUACS)", "m"),
    ("u_curr", "Surface U Current", "m/s"),
    ("v_curr", "Surface V Current", "m/s"),
    ("u_wind", "Surface U Wind", "m/s"),
    ("v_wind", "Surface V Wind", "m/s"),
]
