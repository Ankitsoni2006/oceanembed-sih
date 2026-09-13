from dataclasses import dataclass

@dataclass
class SurfaceDatasetConfig:
    name: str
    source: str
    variable_name: str
    native_resolution_deg: float
    temporal_resolution: str
    units: str
    expected_lon_name: str = 'lon'
    expected_lat_name: str = 'lat'
    
# Do not assume the data is downloaded. 
# This is the formal specification to align the actual data once downloaded.
SURFACE_INPUTS = {
    'SST': SurfaceDatasetConfig(
        name='SST',
        source='Copernicus/OSTIA',
        variable_name='analysed_sst',
        native_resolution_deg=0.05,
        temporal_resolution='daily',
        units='Kelvin'
    ),
    'SSS': SurfaceDatasetConfig(
        name='SSS',
        source='Copernicus/SMAP',
        variable_name='sos',
        native_resolution_deg=0.25,
        temporal_resolution='daily',
        units='psu'
    ),
    'SSH': SurfaceDatasetConfig(
        name='SSH_SLA',
        source='Copernicus/AVISO',
        variable_name='sla',
        native_resolution_deg=0.25,
        temporal_resolution='daily',
        units='meters'
    ),
    'U_CURRENT': SurfaceDatasetConfig(
        name='Surface Eastward Current',
        source='Copernicus/GlobCurrent',
        variable_name='uo',
        native_resolution_deg=0.25,
        temporal_resolution='daily',
        units='m/s'
    ),
    'V_CURRENT': SurfaceDatasetConfig(
        name='Surface Northward Current',
        source='Copernicus/GlobCurrent',
        variable_name='vo',
        native_resolution_deg=0.25,
        temporal_resolution='daily',
        units='m/s'
    ),
    'U_WIND': SurfaceDatasetConfig(
        name='Eastward Wind',
        source='Copernicus/ERA5',
        variable_name='u10',
        native_resolution_deg=0.25,
        temporal_resolution='hourly_to_daily',
        units='m/s'
    ),
    'V_WIND': SurfaceDatasetConfig(
        name='Northward Wind',
        source='Copernicus/ERA5',
        variable_name='v10',
        native_resolution_deg=0.25,
        temporal_resolution='hourly_to_daily',
        units='m/s'
    )
}
