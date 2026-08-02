"""
Process ERA5 meteorological data for Delft3D model domain
Run with 'personal'

This script:
1. Reads Delft3D grid (UTM10N) and converts extent to WGS84 lat/lon
2. Loads ERA5 data (2000-2024) and extracts domain subset
3. Interpolates to regular UTM grid
4. Creates CF-compliant NetCDF files for each variable

Variables extracted:
- air_pressure_fixed_height: Mean sea level pressure (Pa)
- eastward_wind: 10m U wind component (m/s)
- northward_wind: 10m V wind component (m/s)
- air_temperature: 2m temperature (K)
- dew_point_temperature: 2m dewpoint (K)
- cloud_area_fraction: Total cloud cover (0-1)
- surface_solar_radiation: Surface solar radiation (W/m²)
- surface_thermal_radiation: Surface thermal radiation (W/m²)
- surface_sensible_heat_flux: Surface sensible heat flux (W/m²)
- precipitation: Total precipitation (m)
- evaporation: Evaporation (m)
"""

import numpy as np
import xarray as xr
from pathlib import Path
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
import glob
import warnings

# Suppress FutureWarning from xarray concat
warnings.filterwarnings('ignore', category=FutureWarning, module='xarray')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths
ERA5_DIR = Path(r'm:\emeryville_crescent\01_data\era5_meteo_v2')
GRID_FILE = Path(r'm:\emeryville_crescent\03_model_setup\version001\wave\01_overall.grd')

# File format: 'grib' or 'netcdf'
INPUT_FORMAT = 'grib'

# GRIB file templates (used if INPUT_FORMAT = 'grib')
GRIB_INSTANT_PATTERN = 'ERA5_meteo_{year}.grib'
GRIB_ACCUM_PATTERN = 'ERA5_meteo_{year}.grib'

# NetCDF file pattern (used if INPUT_FORMAT = 'netcdf')
NETCDF_PATTERN = 'ERA5_meteo_{year}.nc'

# Output directory
OUTPUT_DIR = Path(r'm:\emeryville_crescent\01_data\era5_meteo_v2\processed_v2')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Time range
YEAR_START = 1940
YEAR_END = 2026

# Variables to extract (mapped from cfgrib short variable names)
# cfgrib reads GRIB files with these short names
VARIABLES = {
    # Instantaneous variables (use 'time' dimension)
    'msl': {
        'output_name': 'air_pressure_fixed_height',
        'long_name': 'Mean sea level pressure',
        'standard_name': 'air_pressure',
        'units': 'Pa',
        'grib_file': 'instant',
        'grib_param_name': 'Mean_sea_level_pressure'
    },
    'u10': {
        'output_name': 'eastward_wind',
        'long_name': '10 metre U wind component',
        'standard_name': 'eastward_wind',
        'units': 'm s**-1',
        'grib_file': 'instant',
        'grib_param_name': 'N10_metre_U_wind_component'
    },
    'v10': {
        'output_name': 'northward_wind',
        'long_name': '10 metre V wind component',
        'standard_name': 'northward_wind',
        'units': 'm s**-1',
        'grib_file': 'instant',
        'grib_param_name': 'N10_metre_V_wind_component'
    },
    't2m': {
        'output_name': 'air_temperature',
        'long_name': '2 metre temperature',
        'standard_name': 'air_temperature',
        'units': 'K',
        'grib_file': 'instant',
        'grib_param_name': 'N2_metre_temperature'
    },
    'd2m': {
        'output_name': 'dew_point_temperature',
        'long_name': '2 metre dewpoint temperature',
        'standard_name': 'dew_point_temperature',
        'units': 'K',
        'grib_file': 'instant',
        'grib_param_name': 'N2_metre_dewpoint_temperature'
    },
    'tcc': {
        'output_name': 'cloud_area_fraction',
        'long_name': 'Total cloud cover',
        'standard_name': 'cloud_area_fraction',
        'units': '1',
        'grib_file': 'instant',
        'grib_param_name': 'Total_cloud_cover'
    },
    # Accumulated variables (use time1, time2, etc. dimensions)
    'ssr': {
        'output_name': 'surface_solar_radiation',
        'long_name': 'Surface solar radiation (solarradiation)',
        'standard_name': 'surface_downwelling_shortwave_flux_in_air',
        'units': 'W m**-2',
        'grib_file': 'accum',
        'grib_param_name': 'Surface_solar_radiation',
        'time_dim': 'time2'
    },
    'str': {
        'output_name': 'surface_thermal_radiation',
        'long_name': 'Surface thermal radiation (longwaveradiation)',
        'standard_name': 'surface_net_thermal_radiation',
        'units': 'W m**-2',
        'grib_file': 'accum',
        'grib_param_name': 'Surface_thermal_radiation',
        'time_dim': 'time5'
    },
    'sshf': {
        'output_name': 'surface_sensible_heat_flux',
        'long_name': 'Surface sensible heat flux',
        'standard_name': 'surface_upward_sensible_heat_flux',
        'units': 'W m**-2',
        'grib_file': 'accum',
        'grib_param_name': 'Surface_sensible_heat_flux',
        'time_dim': 'time4'
    },
    'tp': {
        'output_name': 'precipitation',
        'long_name': 'Total precipitation',
        'standard_name': 'precipitation_amount',
        'units': 'mm hr**-1',  # converted from accumulated m → mm hr**-1 on output
        'grib_file': 'accum',
        'grib_param_name': 'Total_precipitation',
        'time_dim': 'time1'
    },
    'e': {
        'output_name': 'evaporation',
        'long_name': 'Evaporation',
        'standard_name': 'water_evaporation_amount',
        'units': 'm',
        'grib_file': 'accum',
        'grib_param_name': 'Evaporation',
        'time_dim': 'time3'
    }
}

# Coordinate systems
CRS_UTM10N  = 'EPSG:32610'  # UTM Zone 10N
CRS_WGS84   = 'EPSG:4326'     # WGS84 lat/lon

# Output coordinate system (choose 'UTM' or 'WGS84')
OUTPUT_CRS  = 'UTM'  # Set to 'WGS84' for lat/lon output

# UTM grid settings
UTM_GRID_SPACING = 25000  # 25 km (approximately ERA5 resolution)
UTM_BUFFER = 50000  # 50 km buffer around Delft3D domain

# Processing control
PROCESS_VARIABLES = [
    # Instantaneous variables (from instant GRIB files)
    'msl',   # Mean sea level pressure
    'u10',   # 10m U wind component
    'v10',   # 10m V wind component
    't2m',   # 2m temperature
    'd2m',   # 2m dewpoint temperature
    'tcc',   # Total cloud cover
    # Accumulated variables (from accum GRIB files)
    'ssr',   # Surface solar radiation
    'str',   # Surface thermal radiation
    'sshf',  # Surface sensible heat flux
    'tp',    # Total precipitation
    'e'      # Evaporation
]
# List of variables to process, or 'all' for all variables
# cfgrib short names: msl, u10, v10, t2m, d2m, tcc, ssr, str, sshf, tp, e

# ============================================================================
# FUNCTIONS
# ============================================================================

def accum_to_flux_Wm2(da):
    """
    Convert accumulated energy (J m-2) to flux (W m-2)

    ERA5 radiation variables (ssr, strd) are provided as accumulated energy
    over the output interval. This function converts them to instantaneous flux.

    Parameters:
    -----------
    da : xarray.DataArray
        DataArray with accumulated energy in J m-2

    Returns:
    --------
    da_flux : xarray.DataArray
        DataArray with flux in W m-2
    """
    # Compute accumulation interval from time coordinate
    if 'time' in da.dims and len(da.time) > 1:
        # Calculate time differences and take median
        time_diffs = np.diff(da.time.values).astype('timedelta64[s]').astype(float)
        dt_seconds = np.median(time_diffs)
        print(f"    Computed accumulation interval: {dt_seconds:.0f} seconds")
    else:
        # Default to 1 hour if time dimension is missing or has only one value
        dt_seconds = 3600.0
        print(f"    Using default accumulation interval: {dt_seconds:.0f} seconds")

    # Convert J m-2 to W m-2 by dividing by seconds
    da_flux = da / dt_seconds

    # Update units attribute
    da_flux.attrs = da.attrs.copy()
    da_flux.attrs['units'] = 'W m**-2'

    return da_flux


def accum_m_to_mm_per_hr(da):
    """
    Convert accumulated precipitation from m (per timestep) to mm hr**-1.

    ERA5 total precipitation (tp) is stored as accumulated metres over each
    output interval (typically 1 hour for hourly ERA5).  CONUS404 rainfall is
    in mm hr**-1, so we need to:
        1. Divide by the accumulation interval in hours  (de-accumulate to rate)
        2. Multiply by 1000  (m → mm)

    For the standard 1-hour ERA5 output the result is simply tp × 1000.

    Parameters:
    -----------
    da : xarray.DataArray
        DataArray with accumulated precipitation in m

    Returns:
    --------
    da_rate : xarray.DataArray
        DataArray with precipitation rate in mm hr**-1
    """
    if 'time' in da.dims and len(da.time) > 1:
        time_diffs = np.diff(da.time.values).astype('timedelta64[s]').astype(float)
        dt_seconds = np.median(time_diffs)
        print(f"    Computed accumulation interval: {dt_seconds:.0f} seconds")
    else:
        dt_seconds = 3600.0
        print(f"    Using default accumulation interval: {dt_seconds:.0f} seconds")

    dt_hours = dt_seconds / 3600.0

    # m / dt_hours → m hr**-1 → mm hr**-1  (× 1000)
    da_rate = da * (1000.0 / dt_hours)

    da_rate.attrs = da.attrs.copy()
    da_rate.attrs['units'] = 'mm hr**-1'
    print(f"    Conversion: tp [m / {dt_hours:.1f} hr] × 1000 → mm hr**-1  "
          f"(factor = {1000.0 / dt_hours:.1f})")

    return da_rate

def read_delft3d_grid(grid_file):
    """
    Read Delft3D grid file and extract x,y coordinates
    
    Delft3D .grd format structure:
    - Line 0: Coordinate system
    - Line 1: n_cols n_rows
    - Line 2: zeros
    - Lines 3+: First n_rows ETA blocks contain X coordinates
              Next n_rows ETA blocks contain Y coordinates
    
    Parameters:
    -----------
    grid_file : Path
        Path to .grd file
        
    Returns:
    --------
    x : ndarray (n_rows, n_cols)
        X coordinates in grid CRS
    y : ndarray (n_rows, n_cols)
        Y coordinates in grid CRS
    """
    with open(grid_file, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    coord_system = lines[0].strip()
    dims = [int(x) for x in lines[1].split()]
    n_cols, n_rows = dims[0], dims[1]
    
    print(f"Grid dimensions: {n_cols} columns x {n_rows} rows")
    print(f"Coordinate system: {coord_system}")
    
    # Initialize arrays
    x = np.zeros((n_rows, n_cols))
    y = np.zeros((n_rows, n_cols))
    
    # Find all ETA blocks
    eta_blocks = []
    for i, line in enumerate(lines):
        if line.strip().startswith('ETA='):
            eta_blocks.append(i)
    
    print(f"Found {len(eta_blocks)} ETA blocks (expecting {n_rows * 2})")
    
    if len(eta_blocks) < n_rows * 2:
        raise ValueError(f"Expected {n_rows * 2} ETA blocks but found {len(eta_blocks)}")
    
    # Read X coordinates (first n_rows blocks)
    for row in range(n_rows):
        block_start = eta_blocks[row]
        block_end = eta_blocks[row + 1] if row + 1 < len(eta_blocks) else len(lines)
        
        # Collect all values in this block (including values on ETA= line after the row number)
        values = []
        for line_idx in range(block_start, block_end):
            line = lines[line_idx].strip()
            if line.startswith('ETA='):
                # Extract values after "ETA= N" on the same line
                parts = line.split(maxsplit=2)  # Split into ['ETA=', 'N', 'values...']
                if len(parts) > 2:
                    try:
                        values.extend([float(val) for val in parts[2].split()])
                    except ValueError:
                        pass
            else:
                try:
                    values.extend([float(val) for val in line.split()])
                except ValueError:
                    pass
        
        if len(values) >= n_cols:
            x[row, :] = np.array(values[:n_cols])
        else:
            raise ValueError(f"X row {row}: expected {n_cols} values, got {len(values)}")
    
    # Read Y coordinates (second n_rows blocks)
    for row in range(n_rows):
        block_idx = n_rows + row  # Offset by n_rows to get Y blocks
        block_start = eta_blocks[block_idx]
        block_end = eta_blocks[block_idx + 1] if block_idx + 1 < len(eta_blocks) else len(lines)
        
        # Collect all values in this block (including values on ETA= line after the row number)
        values = []
        for line_idx in range(block_start, block_end):
            line = lines[line_idx].strip()
            if line.startswith('ETA='):
                # Extract values after "ETA= N" on the same line
                parts = line.split(maxsplit=2)  # Split into ['ETA=', 'N', 'values...']
                if len(parts) > 2:
                    try:
                        values.extend([float(val) for val in parts[2].split()])
                    except ValueError:
                        pass
            else:
                try:
                    values.extend([float(val) for val in line.split()])
                except ValueError:
                    pass
        
        if len(values) >= n_cols:
            y[row, :] = np.array(values[:n_cols])
        else:
            raise ValueError(f"Y row {row}: expected {n_cols} values, got {len(values)}")
    
    print(f"Successfully read grid:")
    print(f"  X range: [{x.min():.0f}, {x.max():.0f}] m")
    print(f"  Y range: [{y.min():.0f}, {y.max():.0f}] m")
    
    return x, y

def get_grid_extent_latlon(x_utm, y_utm, crs_from=CRS_UTM10N, crs_to=CRS_WGS84):
    """
    Convert grid extent from UTM to lat/lon
    
    Parameters:
    -----------
    x_utm : ndarray
        X coordinates in UTM
    y_utm : ndarray
        Y coordinates in UTM
    crs_from : str
        Source CRS (default: UTM10N)
    crs_to : str
        Target CRS (default: WGS84)
        
    Returns:
    --------
    lon_min, lon_max, lat_min, lat_max : float
        Bounding box in lat/lon
    """
    # Create transformer
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    
    # Get bounds in UTM
    x_min, x_max = np.min(x_utm), np.max(x_utm)
    y_min, y_max = np.min(y_utm), np.max(y_utm)
    
    print(f"\nUTM10N extent:")
    print(f"  X: {x_min:.0f} to {x_max:.0f} m")
    print(f"  Y: {y_min:.0f} to {y_max:.0f} m")
    
    # Transform corner points to lat/lon
    corners_x = [x_min, x_max, x_min, x_max]
    corners_y = [y_min, y_min, y_max, y_max]
    
    lons, lats = transformer.transform(corners_x, corners_y)
    
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    
    print(f"\nWGS84 extent:")
    print(f"  Longitude: {lon_min:.4f}° to {lon_max:.4f}°")
    print(f"  Latitude: {lat_min:.4f}° to {lat_max:.4f}°")
    
    # Add buffer (0.5 degrees)
    buffer = 0.5
    lon_min -= buffer
    lon_max += buffer
    lat_min -= buffer
    lat_max += buffer
    
    print(f"\nWith {buffer}° buffer:")
    print(f"  Longitude: {lon_min:.4f}° to {lon_max:.4f}°")
    print(f"  Latitude: {lat_min:.4f}° to {lat_max:.4f}°")
    
    return lon_min, lon_max, lat_min, lat_max

def create_utm_grid(x_utm, y_utm, spacing=UTM_GRID_SPACING, buffer=UTM_BUFFER):
    """
    Create a regular UTM grid based on Delft3D extent
    
    Parameters:
    -----------
    x_utm : ndarray
        X coordinates in UTM from Delft3D grid
    y_utm : ndarray
        Y coordinates in UTM from Delft3D grid
    spacing : float
        Grid spacing in meters (default: 25 km)
    buffer : float
        Buffer around domain in meters (default: 50 km)
        
    Returns:
    --------
    x_grid : ndarray (1D)
        X coordinates of UTM grid
    y_grid : ndarray (1D)
        Y coordinates of UTM grid
    """
    # Get bounds with buffer
    x_min = np.min(x_utm) - buffer
    x_max = np.max(x_utm) + buffer
    y_min = np.min(y_utm) - buffer
    y_max = np.max(y_utm) + buffer
    
    # Create regular grid
    x_grid = np.arange(x_min, x_max + spacing, spacing)
    y_grid = np.arange(y_min, y_max + spacing, spacing)
    
    print(f"\nUTM grid created:")
    print(f"  X: {x_min/1000:.1f} to {x_max/1000:.1f} km ({len(x_grid)} points)")
    print(f"  Y: {y_min/1000:.1f} to {y_max/1000:.1f} km ({len(y_grid)} points)")
    print(f"  Spacing: {spacing/1000:.1f} km")
    print(f"  Total grid points: {len(x_grid)} x {len(y_grid)} = {len(x_grid)*len(y_grid)}")
    
    return x_grid, y_grid

def _build_bilinear_weights(lat_src, lon_src, lat_tgt, lon_tgt):
    """Precompute 4-corner indices + bilinear weights for interpolation from a
    regular (lat, lon) source grid to a set of target points. Source coordinates
    must be monotone increasing (flip ERA5 latitude before calling).

    Returns: i0, j0 (int arrays, lower-left corner indices) and w00, w01, w10, w11
    (float arrays, weights at corners (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1))."""
    i0 = np.clip(np.searchsorted(lat_src, lat_tgt) - 1, 0, len(lat_src) - 2)
    j0 = np.clip(np.searchsorted(lon_src, lon_tgt) - 1, 0, len(lon_src) - 2)
    alpha = (lat_tgt - lat_src[i0]) / (lat_src[i0 + 1] - lat_src[i0])
    beta  = (lon_tgt - lon_src[j0]) / (lon_src[j0 + 1] - lon_src[j0])
    w00 = (1 - alpha) * (1 - beta)
    w01 = (1 - alpha) * beta
    w10 = alpha       * (1 - beta)
    w11 = alpha       * beta
    return i0, j0, w00, w01, w10, w11


def _apply_bilinear(data, i0, j0, w00, w01, w10, w11, out_shape):
    """Apply precomputed bilinear weights. data: (..., n_lat, n_lon), out: (..., *out_shape).
    Vectorised over arbitrary leading dimensions — single numpy call per variable,
    no per-timestep Python loop."""
    interp = (
        w00 * data[..., i0,     j0]     +
        w01 * data[..., i0,     j0 + 1] +
        w10 * data[..., i0 + 1, j0]     +
        w11 * data[..., i0 + 1, j0 + 1]
    )
    return interp.reshape(data.shape[:-2] + out_shape)


def interpolate_era5_to_utm(ds_era5, x_utm_grid, y_utm_grid, variables_to_process=None, crs_from=CRS_WGS84, crs_to=CRS_UTM10N):
    """
    Interpolate ERA5 data from lat/lon to UTM grid
    
    Parameters:
    -----------
    ds_era5 : xarray.Dataset
        ERA5 dataset in lat/lon
    x_utm_grid : ndarray (1D)
        Target X coordinates in UTM
    y_utm_grid : ndarray (1D)
        Target Y coordinates in UTM
    variables_to_process : list, optional
        List of variable names to interpolate. If None, process all variables.
    crs_from : str
        Source CRS (default: WGS84)
    crs_to : str
        Target CRS (default: UTM10N)
        
    Returns:
    --------
    ds_utm : xarray.Dataset
        Interpolated dataset on UTM grid
    """
    print("\nInterpolating ERA5 to UTM grid...")
    
    # Get coordinate names
    if 'longitude' in ds_era5.coords:
        lon_coord = 'longitude'
        lat_coord = 'latitude'
    else:
        lon_coord = 'lon'
        lat_coord = 'lat'
    
    # Create 2D meshgrid for target UTM coordinates
    X_utm, Y_utm = np.meshgrid(x_utm_grid, y_utm_grid)
    
    # Transform UTM grid to lat/lon for interpolation
    transformer = Transformer.from_crs(crs_to, crs_from, always_xy=True)
    LON_target, LAT_target = transformer.transform(X_utm.ravel(), Y_utm.ravel())
    LON_target = LON_target.reshape(X_utm.shape)
    LAT_target = LAT_target.reshape(Y_utm.shape)
    
    # Get ERA5 coordinates
    lon_era5 = ds_era5[lon_coord].values
    lat_era5 = ds_era5[lat_coord].values

    # RegularGridInterpolator requires strictly ascending coordinates.
    # ERA5 latitude is typically stored descending (90° → -90°), so flip if needed.
    if lat_era5[0] > lat_era5[-1]:
        lat_era5 = lat_era5[::-1]
        flip_lat = True
        print("  ERA5 latitude is descending — flipping to ascending order for interpolation")
    else:
        flip_lat = False

    # Check if ERA5 uses 0-360 longitude convention and adjust target coordinates
    if lon_era5.max() > 180:
        print("  Adjusting target coordinates to 0-360° convention...")
        LON_target = np.where(LON_target < 0, LON_target + 360, LON_target)

    # Precompute bilinear interpolation weights ONCE — source + target grids are
    # constant across the time axis, so the per-target-cell weights and corner
    # indices are identical for every timestep. Apply via vectorised numpy
    # fancy-indexing instead of a per-timestep RegularGridInterpolator loop.
    i0, j0, w00, w01, w10, w11 = _build_bilinear_weights(
        lat_era5, lon_era5, LAT_target.ravel(), LON_target.ravel()
    )
    out_shape = Y_utm.shape
    print(f"  Precomputed bilinear weights: {len(i0)} target cells, {len(lat_era5)}×{len(lon_era5)} source grid")

    # Find all time-like dimensions in the dataset
    time_dims = [dim for dim in ds_era5.dims if 'time' in dim.lower()]
    print(f"  Time dimensions found: {time_dims}")
    
    # Determine which variables to interpolate
    if variables_to_process is None:
        vars_to_interpolate = list(ds_era5.data_vars)
    else:
        vars_to_interpolate = [v for v in variables_to_process if v in ds_era5.data_vars]
        missing_vars = [v for v in variables_to_process if v not in ds_era5.data_vars]
        if missing_vars:
            print(f"  Warning: Variables not found in dataset: {missing_vars}")
    
    print(f"  Interpolating {len(vars_to_interpolate)} variable(s): {vars_to_interpolate}")
    
    # Create output dataset - we'll add coordinates as we process variables
    ds_utm = xr.Dataset(
        coords={
            'x': x_utm_grid,
            'y': y_utm_grid
        }
    )
    
    # Add spatial coordinate metadata
    ds_utm['x'].attrs = {'units': 'm', 'long_name': 'Easting (UTM10N)', 'standard_name': 'projection_x_coordinate', 'axis': 'X'}
    ds_utm['y'].attrs = {'units': 'm', 'long_name': 'Northing (UTM10N)', 'standard_name': 'projection_y_coordinate', 'axis': 'Y'}
    
    # Interpolate each variable
    for var_name in vars_to_interpolate:
        print(f"  Interpolating {var_name}...")
        
        # Find the time dimension for this variable
        var_dims = ds_era5[var_name].dims
        var_time_dim = None
        for dim in var_dims:
            if 'time' in dim.lower():
                var_time_dim = dim
                break
        
        # Handle time dimension
        if var_time_dim is not None:
            print(f"    Using time dimension: {var_time_dim}")
            
            # Get time values for this variable
            time_values = ds_era5[var_time_dim].values
            n_times = len(time_values)
            
            # Add time coordinate to output if not already present
            # Use 'time' as standardized output dimension name
            if 'time' not in ds_utm.coords:
                ds_utm = ds_utm.assign_coords(time=('time', time_values))
                ds_utm['time'].attrs = {
                    'long_name': 'time',
                    'standard_name': 'time',
                    'axis': 'T'
                }
            
            # Pull the full (time, lat, lon) array. Explicit transpose guarantees
            # the leading dim is the variable's time dim — accum variables often
            # carry auxiliary coords (step, valid_time) that would otherwise leave
            # `.values` in an unexpected order. Flip lat axis to match the
            # ascending-source convention assumed by the precomputed weights.
            var_data_np = ds_era5[var_name].transpose(var_time_dim, lat_coord, lon_coord).values
            if flip_lat:
                var_data_np = var_data_np[..., ::-1, :]

            # One-shot NaN fill across the whole array (rare for ERA5 surface fields;
            # this branch usually doesn't fire). Done once per variable, not per timestep.
            if np.any(np.isnan(var_data_np)):
                mask = np.isnan(var_data_np)
                if not np.all(mask):
                    var_data_np = np.copy(var_data_np)
                    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
                    var_data_np[mask] = var_data_np[tuple(idx[..., mask])]

            # Vectorised bilinear interpolation — single numpy call over all timesteps.
            interpolated = _apply_bilinear(
                var_data_np, i0, j0, w00, w01, w10, w11, out_shape
            )
            print(f"    Interpolated {n_times} timesteps in one vectorised call (shape {interpolated.shape})")

            # Add to dataset with standardized 'time' dimension name
            ds_utm[var_name] = (['time', 'y', 'x'], interpolated)
        else:
            # No time dimension — apply weights to a single (lat, lon) slice.
            data_slice = ds_era5[var_name].transpose(lat_coord, lon_coord).values
            if flip_lat:
                data_slice = data_slice[::-1, :]

            if np.any(np.isnan(data_slice)):
                mask = np.isnan(data_slice)
                if not np.all(mask):
                    data_slice = np.copy(data_slice)
                    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
                    data_slice[mask] = data_slice[tuple(idx[:, mask])]

            interpolated = _apply_bilinear(
                data_slice, i0, j0, w00, w01, w10, w11, out_shape
            )
            ds_utm[var_name] = (['y', 'x'], interpolated)
        
        # Copy attributes
        ds_utm[var_name].attrs = ds_era5[var_name].attrs.copy()
    
    # Add CRS information
    ds_utm.attrs['crs'] = CRS_UTM10N
    ds_utm.attrs['grid_mapping_name'] = 'transverse_mercator'
    
    print("  Interpolation complete!")
    
    return ds_utm

def load_era5_netcdf(era5_dir, year_start, year_end):
    """
    Load ERA5 data from NetCDF files for specified years
    
    Handles NetCDF files with multiple time dimensions (time, time1, time2, etc.)
    where different variables use different time coordinates.
    
    Parameters:
    -----------
    era5_dir : Path
        Directory containing ERA5 NetCDF files
    year_start : int
        Start year
    year_end : int
        End year
        
    Returns:
    --------
    ds : xarray.Dataset
        Combined ERA5 dataset with all variables
    """
    netcdf_files = []
    
    for year in range(year_start, year_end + 1):
        pattern = era5_dir / NETCDF_PATTERN.format(year=year)
        matches = glob.glob(str(pattern))
        
        if matches:
            netcdf_files.extend(matches)
            print(f"Found: {matches[0]}")
        else:
            print(f"Warning: No NetCDF file found for year {year}")
    
    if not netcdf_files:
        raise FileNotFoundError(f"No ERA5 NetCDF files found in {era5_dir}")
    
    # Load NetCDF files
    print(f"\nLoading {len(netcdf_files)} NetCDF files...")
    datasets = []
    
    for f in netcdf_files:
        try:
            ds = xr.open_dataset(f)
            
            # Squeeze out ensemble dimension if present
            if 'ens' in ds.dims:
                ds = ds.isel(ens=0)
                
            # Print info about dimensions and variables
            print(f"  Loaded: {Path(f).name}")
            print(f"    Variables: {list(ds.data_vars)}")
            print(f"    Dimensions: {dict(ds.dims)}")
            
            datasets.append(ds)
            
        except Exception as e:
            print(f"  Error loading {Path(f).name}: {e}")
    
    if not datasets:
        raise RuntimeError("Failed to load any NetCDF files")
    
    # Concatenate along appropriate time dimensions
    if len(datasets) == 1:
        ds_combined = datasets[0]
    else:
        # For NetCDF files with multiple time dimensions, we need to handle concatenation carefully
        # Find all time-like dimensions
        time_dims = set()
        for ds in datasets:
            for dim in ds.dims:
                if 'time' in dim.lower():
                    time_dims.add(dim)
        
        print(f"  Time dimensions found: {time_dims}")
        
        # Concatenate along each time dimension
        ds_combined = xr.concat(datasets, dim='time', data_vars='minimal', coords='minimal', 
                               compat='override', join='outer')
    
    print(f"\nLoaded ERA5 data:")
    print(f"  Variables: {list(ds_combined.data_vars)}")
    print(f"  Dimensions: {dict(ds_combined.dims)}")
    
    return ds_combined

def load_era5_data(era5_dir, year_start, year_end):
    """
    Load ERA5 data from GRIB files for specified years
    
    Parameters:
    -----------
    era5_dir : Path
        Directory containing ERA5 GRIB files
    year_start : int
        Start year
    year_end : int
        End year
        
    Returns:
    --------
    ds_instant : xarray.Dataset
        Combined ERA5 instantaneous dataset
    ds_accum : xarray.Dataset
        Combined ERA5 accumulated dataset
    """
    # Find all ERA5 GRIB files for the year range
    instant_files = []
    accum_files = []
    
    for year in range(year_start, year_end + 1):
        instant_pattern = era5_dir / GRIB_INSTANT_PATTERN.format(year=year)
        accum_pattern = era5_dir / GRIB_ACCUM_PATTERN.format(year=year)
        
        instant_matches = glob.glob(str(instant_pattern))
        accum_matches = glob.glob(str(accum_pattern))
        
        if instant_matches:
            instant_files.extend(instant_matches)
            print(f"Found instant: {instant_matches[0]}")
        else:
            print(f"Warning: No instant file found for year {year}")
            
        if accum_matches:
            accum_files.extend(accum_matches)
            print(f"Found accum: {accum_matches[0]}")
        else:
            print(f"Warning: No accum file found for year {year}")
    
    if not instant_files and not accum_files:
        raise FileNotFoundError(f"No ERA5 GRIB files found in {era5_dir}")
    
    # Load instantaneous files
    ds_instant = None
    if instant_files:
        print(f"\nLoading {len(instant_files)} instantaneous GRIB files...")
        # Load files one at a time to avoid GDAL issues
        datasets = []
        for f in instant_files:
            try:
                ds = xr.open_dataset(
                    f,
                    engine='cfgrib',
                    backend_kwargs={'errors': 'ignore'}
                )
                # Squeeze out ensemble dimension if present
                if 'ens' in ds.dims:
                    ds = ds.isel(ens=0)
                datasets.append(ds)
                print(f"  Loaded: {Path(f).name}")
            except Exception as e:
                print(f"  Error loading {Path(f).name}: {e}")
        
        if datasets:
            ds_instant = xr.concat(datasets, dim='time', data_vars='minimal', coords='minimal', compat='override', join='outer')
            print(f"Loaded instantaneous data from {ds_instant.time.min().values} to {ds_instant.time.max().values}")
            print(f"Variables: {list(ds_instant.data_vars)}")
        else:
            print("Failed to load any instantaneous files")
    
    # Load accumulated files
    ds_accum = None
    if accum_files:
        print(f"\nLoading {len(accum_files)} accumulated GRIB files...")
        # GRIB files with multiple time dimensions need special handling
        # Open the file multiple times to get each variable separately
        all_datasets = []
        
        for f in accum_files:
            print(f"  Loading: {Path(f).name}")
            
            # Extract year from filename to filter timestamps
            filename = Path(f).name
            import re
            year_match = re.search(r'(\d{4})', filename)
            if year_match:
                file_year = int(year_match.group(1))
            else:
                file_year = None
            
            file_datasets = []
            
            # Try to open and get all variables
            try:
                # First, check what variables are available
                ds_check = xr.open_dataset(f, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
                available_vars = list(ds_check.data_vars)
                print(f"    Found variables: {available_vars}")
                ds_check.close()
                
                # For accumulated files, try to open each variable we need separately
                # Short names: SSR, STR, SSHF, TP, E
                target_vars = ['ssr', 'str', 'sshf', 'tp', 'e']  # All accumulated variables
                
                for var_short in target_vars:
                    try:
                        # Try to open with filter for this specific variable
                        ds_var = xr.open_dataset(
                            f,
                            engine='cfgrib',
                            backend_kwargs={
                                'filter_by_keys': {'shortName': var_short},
                                'errors': 'ignore'
                            }
                        )
                        
                        if var_short in ds_var.data_vars or var_short.upper() in [v.lower() for v in ds_var.data_vars]:
                            print(f"    Processing {var_short}...")
                            
                            # Squeeze out ensemble dimension if present
                            if 'ens' in ds_var.dims:
                                ds_var = ds_var.isel(ens=0)
                            
                            # Debug: print dimensions before processing
                            print(f"      Dims before: {dict(ds_var.dims)}")
                            
                            # Handle step dimension for accumulated variables
                            # ERA5 accumulated data has dimensions (time, step, lat, lon)
                            # where valid_time is (time, step) giving the actual timestamp for each accumulation
                            if 'step' in ds_var.dims and 'valid_time' in ds_var.coords:
                                print(f"      Reshaping from (time={len(ds_var.time)}, step={len(ds_var.step)}) to 1D time")
                                
                                # Stack time and step dimensions into a single time dimension
                                # This flattens the 2D time structure into 1D
                                ds_var = ds_var.stack(new_time=('time', 'step'))
                                
                                # Get valid_time values (also 2D, needs to be flattened)
                                valid_times = ds_var['valid_time'].values
                                
                                # Replace new_time coordinate with actual valid_time values
                                ds_var = ds_var.assign_coords(new_time=valid_times)
                                
                                # Rename new_time to time
                                ds_var = ds_var.rename({'new_time': 'time'})
                                
                                # Drop old coordinates we don't need
                                ds_var = ds_var.drop_vars(['valid_time', 'step'], errors='ignore')
                                
                                # Sort by time to ensure chronological order
                                ds_var = ds_var.sortby('time')
                                
                                print(f"      Reshaped to time dimension of length {len(ds_var.time)}")
                                
                                # Filter to only include timestamps that belong to the file's year
                                if file_year is not None:
                                    import pandas as pd
                                    time_values = pd.to_datetime(ds_var.time.values)
                                    year_mask = time_values.year == file_year
                                    ds_var = ds_var.isel(time=year_mask)
                                    print(f"      Filtered to {file_year}: {len(ds_var.time)} timesteps remaining")
                                    print(f"      Time range after filter: {ds_var.time.min().values} to {ds_var.time.max().values}")
                                
                            elif 'step' in ds_var.dims:
                                # No valid_time, just select first step
                                ds_var = ds_var.isel(step=0)
                                ds_var = ds_var.drop_vars(['step'], errors='ignore')
                            
                            print(f"      Dims after: {dict(ds_var.dims)}")
                            print(f"      Time range: {ds_var.time.min().values} to {ds_var.time.max().values}")
                            
                            file_datasets.append(ds_var)
                            print(f"    Loaded {var_short}")
                        
                    except Exception as e:
                        # Variable not found or error, skip
                        print(f"    Could not load {var_short}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # If we got datasets for this file, merge them
                if file_datasets:
                    try:
                        if len(file_datasets) == 1:
                            merged_ds = file_datasets[0]
                        else:
                            # Merge all variables on the common time dimension
                            merged_ds = xr.merge(file_datasets, compat='override')
                        all_datasets.append(merged_ds)
                        print(f"    Successfully merged {len(file_datasets)} variable(s)")
                    except Exception as e:
                        print(f"    Error merging datasets: {e}")
                        # Try adding them separately if merge fails
                        all_datasets.extend(file_datasets)
                    
            except Exception as e:
                print(f"    Error loading {Path(f).name}: {e}")
        
        if all_datasets:
            ds_accum = xr.concat(all_datasets, dim='time', data_vars='minimal', coords='minimal', compat='override', join='outer')
            print(f"Loaded accumulated data from {ds_accum.time.min().values} to {ds_accum.time.max().values}")
            print(f"Variables: {list(ds_accum.data_vars)}")
            print(f"Dimensions: {dict(ds_accum.dims)}")
            print(f"Coordinates: {list(ds_accum.coords.keys())}")
        else:
            print("Failed to load any accumulated files")
    
    return ds_instant, ds_accum

def extract_domain(ds, lon_min, lon_max, lat_min, lat_max):
    """
    Extract subset of ERA5 data for specified domain
    
    Parameters:
    -----------
    ds : xarray.Dataset or None
        Full ERA5 dataset
    lon_min, lon_max, lat_min, lat_max : float
        Bounding box in lat/lon
        
    Returns:
    --------
    ds_subset : xarray.Dataset or None
        Subsetted dataset
    """
    if ds is None:
        return None
        
    # Handle longitude wrapping if necessary
    if 'longitude' in ds.coords:
        lon_coord = 'longitude'
        lat_coord = 'latitude'
    elif 'lon' in ds.coords:
        lon_coord = 'lon'
        lat_coord = 'lat'
    else:
        raise ValueError("Cannot find longitude/latitude coordinates")
    
    # Check if ERA5 uses 0-360 longitude convention
    lon_values = ds[lon_coord].values
    if lon_values.max() > 180:
        # ERA5 uses 0-360, convert negative domain bounds to 0-360
        print(f"  ERA5 uses 0-360° longitude convention")
        if lon_min < 0:
            lon_min += 360
        if lon_max < 0:
            lon_max += 360
        print(f"  Adjusted bounds: {lon_min:.2f}° to {lon_max:.2f}°")
    
    # Select domain
    ds_subset = ds.sel(
        {lon_coord: slice(lon_min, lon_max),
         lat_coord: slice(lat_max, lat_min)}  # Latitude often reversed
    )
    
    print(f"\nExtracted domain:")
    print(f"  {lon_coord}: {float(ds_subset[lon_coord].min()):.2f}° to {float(ds_subset[lon_coord].max()):.2f}°")
    print(f"  {lat_coord}: {float(ds_subset[lat_coord].min()):.2f}° to {float(ds_subset[lat_coord].max()):.2f}°")
    print(f"  Grid size: {len(ds_subset[lon_coord])} x {len(ds_subset[lat_coord])}")
    
    return ds_subset

def save_variable_to_netcdf(ds, var_name, output_dir, metadata):
    """
    Save individual variable to NetCDF with CF-compliant format
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing the variable
    var_name : str
        Variable name to extract (ERA5 name)
    output_dir : Path
        Output directory
    metadata : dict
        Variable metadata (output_name, long_name, standard_name, units)
    """
    if var_name not in ds:
        print(f"Warning: Variable '{var_name}' not found in dataset")
        return
    
    # Extract variable and convert to float32
    da = ds[var_name].astype('float32')
    
    # Convert accumulated radiation/heat flux to flux if needed
    # These variables have units "W m-2 s" which means accumulated energy (J/m²)
    accumulated_flux_vars = ['ssr', 'str', 'sshf']
    if var_name in accumulated_flux_vars:
        print(f"  Converting {var_name} from accumulated energy (J m-2) to flux (W m-2)...")
        da = accum_to_flux_Wm2(da)

        # Sanity check: print min/max values
        print(f"  {var_name} flux range: {float(da.min()):.2f} to {float(da.max()):.2f} W m**-2")
        if var_name == 'ssr':
            print(f"    Expected: daytime peaks ~600-1100 W m**-2, nighttime ~0 W m**-2")
        elif var_name == 'str':
            print(f"    Expected: net thermal radiation, can be positive or negative")
        elif var_name == 'sshf':
            print(f"    Expected: sensible heat flux, typically -100 to 400 W m**-2")

    # Convert accumulated precipitation from m to mm hr**-1 to match CONUS404 units
    if var_name == 'tp':
        print(f"  Converting tp from accumulated m to mm hr**-1 (×1000 for hourly ERA5)...")
        da = accum_m_to_mm_per_hr(da)
        print(f"  tp rate range: {float(da.min()):.4f} to {float(da.max()):.4f} mm hr**-1")
        print(f"    Expected: 0 to ~100 mm hr**-1 (extreme convective events)")

    # ---- Physical bounds clipping -----------------------------------------------
    # Bilinear interpolation (ERA5 lat/lon -> UTM) can produce small overshoots near
    # zero or at the edges of the domain.  Clip variables that have hard physical
    # limits.  Evaporation, thermal radiation and sensible heat flux are intentionally
    # left unconstrained because negative values are physically valid for those fields.
    clip_bounds = {
        # var_name : (min,  max)
        'tcc'      : (0.0,  1.0),   # cloud fraction 0-1
        'ssr'      : (0.0,  None),  # downwelling solar >= 0
        'tp'       : (0.0,  None),  # precipitation >= 0
    }
    if var_name in clip_bounds:
        vmin, vmax = clip_bounds[var_name]
        n_before = int((da.values < (vmin if vmin is not None else -1e38)).sum() +
                       (da.values > (vmax if vmax is not None else  1e38)).sum())
        da = da.clip(min=vmin, max=vmax)
        print(f"  Clipped {var_name} to [{vmin}, {vmax}]: {n_before} values corrected")
    # -----------------------------------------------------------------------------

    # Get output variable name
    output_var_name = metadata.get('output_name', var_name)
    
    # Rename variable
    da.name = output_var_name
    
    # Set metadata
    da.attrs = {
        'coordinates': 'y x',
        'long_name': metadata.get('long_name', var_name),
        'standard_name': metadata.get('standard_name', var_name),
        'units': metadata.get('units', '')
    }
    
    # Create output dataset
    ds_out = da.to_dataset()
    
    # Update coordinate attributes based on output CRS
    if OUTPUT_CRS == 'UTM':
        # UTM coordinates (x/y in meters)
        if 'x' in ds_out.coords:
            ds_out['x'].attrs = {
                'units': 'm',
                'long_name': 'x coordinate (UTM10N)',
                'standard_name': 'projection_x_coordinate'
            }
        if 'y' in ds_out.coords:
            ds_out['y'].attrs = {
                'units': 'm',
                'long_name': 'y coordinate (UTM10N)',
                'standard_name': 'projection_y_coordinate'
            }
    else:
        # WGS84 coordinates (lat/lon in degrees)
        if 'x' in ds_out.coords:
            ds_out['x'].attrs = {
                'units': 'degrees_east',
                'long_name': 'longitude'
            }
        if 'y' in ds_out.coords:
            ds_out['y'].attrs = {
                'units': 'degrees_north',
                'long_name': 'latitude'
            }
    
    if 'time' in ds_out.coords:
        # Remove units and calendar from attrs to avoid conflict with encoding
        ds_out['time'].attrs = {
            'long_name': 'time',
            'standard_name': 'time',
            'axis': 'T'
        }
    
    # Spatial coordinates with axis attributes
    if 'x' in ds_out.coords:
        ds_out['x'].attrs['axis'] = 'X'
    if 'y' in ds_out.coords:
        ds_out['y'].attrs['axis'] = 'Y'
    
    # Save with encoding
    crs_suffix = 'UTM' if OUTPUT_CRS == 'UTM' else 'WGS84'
    output_file = output_dir / f'ERA5_{output_var_name}_{YEAR_START}_{YEAR_END}_{crs_suffix}.nc'
    
    # Determine coordinate names and dimensions
    coord_names = list(ds_out.coords.keys())
    
    # Build encoding
    encoding = {}
    
    # Time coordinate encoding
    if 'time' in coord_names:
        time_size = len(ds_out.time)
        time_chunksize = min(1024, time_size)  # Don't exceed actual time dimension size
        encoding['time'] = {
            'units': 'hours since 1900-01-01 00:00:00.0',
            'calendar': 'gregorian',
            'dtype': 'float64',  # Use float64 for time to preserve precision
            'chunksizes': (time_chunksize,),
            'zlib': False
        }
    
    # Spatial coordinates
    for coord in ['x', 'y']:
        if coord in coord_names:
            encoding[coord] = {'dtype': 'float32'}
    
    # Variable encoding
    if 'x' in coord_names and 'y' in coord_names:
        ny, nx = len(ds_out.y), len(ds_out.x)
        if 'time' in ds_out[output_var_name].dims:
            chunk_shape = (1, ny, nx)  # (1, lat, lon) chunking
        else:
            chunk_shape = (ny, nx)
    else:
        chunk_shape = None
    
    encoding[output_var_name] = {
        'dtype': 'float32',
        'zlib': False,  # Deflate = 0
    }
    
    if chunk_shape is not None:
        encoding[output_var_name]['chunksizes'] = chunk_shape
    
    print(f"Saving {output_var_name} to {output_file.name}...")
    ds_out.to_netcdf(output_file, encoding=encoding)
    print(f"  File size: {output_file.stat().st_size / 1e6:.1f} MB")

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main processing function"""
    
    print("="*70)
    print("ERA5 Meteorological Data Processing")
    print("="*70)
    
    # Step 1: Read Delft3D grid
    print("\n1. Reading Delft3D grid...")
    x_utm, y_utm = read_delft3d_grid(GRID_FILE)
    
    # Step 2: Convert extent to lat/lon
    print("\n2. Converting grid extent to WGS84...")
    lon_min, lon_max, lat_min, lat_max = get_grid_extent_latlon(x_utm, y_utm)
    
    # Step 3: Load ERA5 data
    print(f"\n3. Loading ERA5 data (format: {INPUT_FORMAT})...")
    
    if INPUT_FORMAT == 'netcdf':
        # Load from NetCDF files (single dataset with all variables)
        ds_era5 = load_era5_netcdf(ERA5_DIR, YEAR_START, YEAR_END)
        ds_instant = None
        ds_accum = None
        use_single_dataset = True
    else:
        # Load from GRIB files (separate instant and accum datasets)
        ds_instant, ds_accum = load_era5_data(ERA5_DIR, YEAR_START, YEAR_END)
        ds_era5 = None
        use_single_dataset = False
    
    # Step 4: Extract domain
    print("\n4. Extracting model domain...")
    if use_single_dataset:
        ds_era5_subset = extract_domain(ds_era5, lon_min, lon_max, lat_min, lat_max)
    else:
        ds_instant_subset = extract_domain(ds_instant, lon_min, lon_max, lat_min, lat_max)
        ds_accum_subset = extract_domain(ds_accum, lon_min, lon_max, lat_min, lat_max)
    
    # Step 5: Create UTM grid
    print("\n5. Creating UTM grid...")
    x_utm_grid, y_utm_grid = create_utm_grid(x_utm, y_utm)
    
    # Determine which variables to process
    if PROCESS_VARIABLES == 'all':
        vars_to_process = list(VARIABLES.keys())
    else:
        vars_to_process = PROCESS_VARIABLES
    
    print(f"\nVariables to process: {vars_to_process}")
    
    # Step 6: Process each variable from the appropriate dataset
    print("\n6. Interpolating ERA5 to UTM grid and saving...")
    
    for var_name in vars_to_process:
        if var_name not in VARIABLES:
            print(f"Warning: Metadata not found for variable '{var_name}', skipping.")
            continue
            
        metadata = VARIABLES[var_name]
        
        if use_single_dataset:
            # NetCDF format: all variables in single dataset
            ds_source = ds_era5_subset
        else:
            # GRIB format: select appropriate dataset based on variable type
            grib_file_type = metadata.get('grib_file', 'instant')
            if grib_file_type == 'instant':
                ds_source = ds_instant_subset
            else:
                ds_source = ds_accum_subset
            
        if ds_source is None:
            print(f"Warning: No dataset available for '{var_name}', skipping.")
            continue
            
        if var_name not in ds_source:
            print(f"Warning: Variable '{var_name}' not found in dataset, skipping.")
            # Print available variables for debugging
            print(f"  Available variables: {list(ds_source.data_vars)}")
            continue
        
        print(f"\nProcessing {var_name}...")
        
        # Interpolate this variable to UTM grid
        ds_utm = interpolate_era5_to_utm(ds_source, x_utm_grid, y_utm_grid, variables_to_process=[var_name])
        
        # Save to NetCDF
        save_variable_to_netcdf(ds_utm, var_name, OUTPUT_DIR, metadata)
    
    # Close datasets
    if use_single_dataset:
        if ds_era5 is not None:
            ds_era5.close()
    else:
        if ds_instant is not None:
            ds_instant.close()
        if ds_accum is not None:
            ds_accum.close()
    
    print("\n" + "="*70)
    print("Processing complete!")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == '__main__':
    main()
