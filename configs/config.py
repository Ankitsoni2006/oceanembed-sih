from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class GridConfig:
    lat_min: float = 5.0
    lat_max: float = 30.0
    lon_min: float = 45.0
    lon_max: float = 105.0
    resolution: float = 0.25

@dataclass
class DataConfig:
    input_variables: List[str] = field(default_factory=lambda: [
        'sst', 'sss', 'ssh', 'u_current', 'v_current', 'u_wind', 'v_wind'
    ])
    target_depths: List[float] = field(default_factory=lambda: [
        0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
    ])
    # Placeholder periods; exact historical periods depend on dataset overlap verification
    train_years: Tuple[int, int] = (2016, 2020)
    val_years: Tuple[int, int] = (2021, 2021)
    test_years: Tuple[int, int] = (2022, 2022)

@dataclass
class OceanEmbedConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    data: DataConfig = field(default_factory=DataConfig)

config = OceanEmbedConfig()
