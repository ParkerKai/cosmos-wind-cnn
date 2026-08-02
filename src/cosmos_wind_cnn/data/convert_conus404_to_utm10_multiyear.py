"""
Convert CONUS404 wind data to UTM10 grid
Reads Lambert Conformal projection data and interpolates to UTM Zone 10N grid
Processes multiple years of Water Year data and combines them into single output files
Processes both 4km (SFbay) and 1km (BA) resolution datasets
"""

import numpy as np
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta
import os

# %% Conversion functions

def convert_surface_to_msl_pressure(psfc, terrain_height, temperature):
    """
    Convert surface pressure to mean sea level (MSL) pressure.
    
    Uses the barometric formula with standard atmosphere lapse rate.
    
    Parameters:
    -----------
    psfc : ndarray
        Surface pressure in Pa
    terrain_height : ndarray
        Terrain height in meters (can be time-invariant or time-varying)
    temperature : ndarray
        Temperature at 2m in Kelvin (same shape as psfc)
    
    Returns:
    --------
    pmsl : ndarray
        Mean sea level pressure in Pa
    
    Formula:
    --------
    PMSL = PSFC * exp(g * H / (R * T))
    
    where:
    - g = 9.80665 m/s² (gravitational acceleration)
    - H = terrain height (m)
    - R = 287.05 J/(kg·K) (gas constant for dry air)
    - T = temperature (K)
    """
    # Constants
    g = 9.80665  # m/s²
    R = 287.05   # J/(kg·K)
    
    # Calculate MSL pressure
    # If terrain_height has fewer dimensions than psfc, broadcast it
    pmsl = psfc * np.exp(g * terrain_height / (R * temperature))
    
    return pmsl

def convert_accumulated_to_instantaneous_flux(accumulated_data, time_coord):
    """
    Convert accumulated radiation flux (J m-2) to instantaneous flux (W m-2).
    
    CONUS404 radiation variables (ACSWDNB, ACLWDNB) are accumulated since 1979-10-01.
    This function computes the flux by taking differences between consecutive timesteps
    and dividing by the time interval.
    
    Parameters:
    -----------
    accumulated_data : ndarray
        Accumulated flux in J m-2, shape (time, y, x)
    time_coord : ndarray
        Time coordinate in hours since 1900-01-01
    
    Returns:
    --------
    flux_data : ndarray
        Instantaneous flux in W m-2, same shape as input
    
    Notes:
    ------
    - First timestep uses forward difference
    - Subsequent timesteps use backward difference
    - Flux = (accumulated[t] - accumulated[t-1]) / dt_seconds
    - 1 W = 1 J/s
    """
    nt = accumulated_data.shape[0]
    flux_data = np.zeros_like(accumulated_data)
    
    # Compute time intervals in seconds
    # time_coord is in hours, convert to seconds
    time_seconds = time_coord * 3600.0
    
    # First timestep: use forward difference
    if nt > 1:
        dt = time_seconds[1] - time_seconds[0]
        flux_data[0, :, :] = (accumulated_data[1, :, :] - accumulated_data[0, :, :]) / dt
    else:
        # Only one timestep, cannot compute flux - set to zero
        flux_data[0, :, :] = 0.0
    
    # Remaining timesteps: use backward difference
    for t in range(1, nt):
        dt = time_seconds[t] - time_seconds[t-1]
        flux_data[t, :, :] = (accumulated_data[t, :, :] - accumulated_data[t-1, :, :]) / dt
    
    # Handle any negative values (can occur due to resets or errors)
    # Set negative fluxes to zero as they are physically unrealistic
    flux_data = np.maximum(flux_data, 0.0)
    
    return flux_data

# %% Configuration for both datasets

# ========== EXTRAPOLATION SETTING ==========
# Set to True to extrapolate beyond input domain (prevents NaNs)
# Set to False to get NaNs for points outside the input domain
USE_EXTRAPOLATION = True
# ===========================================

datasets = [
    {
        'name': 'BA_1km',
        'input_dir': r'm:\emeryville_crescent\01_data\Conus404_BA_v2',
        'file_pattern': 'conus404BA_SFbay_WY{year}.nc',
        'resolution': 1000,  # 1 km
        'description': '1km resolution Bay Area',
        'wind_vars': {
            'U2D': 'eastward_wind',
            'V2D': 'northward_wind'
        }
    },
    {
        'name': 'SFbay_4km',
        'input_dir': r'm:\emeryville_crescent\01_data\Conus404_SFbay',
        'file_pattern': 'conus404_SFbay_WY{year}.nc',
        'resolution': 4000,  # 4 km
        'description': '4km resolution San Francisco Bay',
        'wind_vars': {
            'U10': 'eastward_wind',
            'V10': 'northward_wind'
        }
    }
]

datasets = [
    {
        'name': 'SFbay_4km',
        'input_dir': r'm:\emeryville_crescent\01_data\Conus404_SFbay_V3',
        'file_pattern': 'conus404_SFbay_WY{year}.nc',
        'resolution': 4000,  # 4 km
        'description': '4km resolution San Francisco Bay',
        'wind_vars': {
            'U10': 'eastward_wind',
            'V10': 'northward_wind',
            'T2': 'air_temperature',
            'TD2': 'dew_point_temperature',
            'PREC_ACC_NC': 'rainfall',
            'PSFC': 'air_pressure_fixed_height',  # Will be converted from surface to MSL
            'ACSWDNB': 'surface_solar_radiation',
            'ACLWDNB': 'surface_thermal_radiation_downwards',
        },
        'pressure_conversion': {
            'surface_pressure': 'PSFC',
            'temperature': 'T2',
            'terrain_height': 'HGT'
        }
    }
]


# Date range to process (Water Years)
# Note: WY2000 = Oct 1999 - Sep 2000
start_wy    = 1979
end_wy      = 2021

# UTM10N output grid extent (will be refined based on input resolution)
utm_x_min = 425942-1000
utm_x_max = 596072+1000
utm_y_min = 4091803-1000
utm_y_max = 4257374+1000

print("="*70)
print("CONUS404 to UTM10N Wind Converter (Multi-Year, Multi-Resolution)")
print("="*70)
print(f"Processing Water Years: {start_wy} to {end_wy}")
print(f"Extrapolation: {'ENABLED' if USE_EXTRAPOLATION else 'DISABLED'}")
print(f"Datasets to process: {len(datasets)}")
for ds in datasets:
    print(f"  - {ds['name']}: {ds['description']} ({ds['resolution']}m)")

# %% Process each dataset
for dataset_idx, dataset in enumerate(datasets):
    print("\n" + "="*70)
    print(f"PROCESSING DATASET {dataset_idx+1}/{len(datasets)}: {dataset['name']}")
    print("="*70)
    
    input_dir = dataset['input_dir']
    file_pattern = dataset['file_pattern']
    grid_resolution = dataset['resolution']
    wind_vars = dataset['wind_vars']  # Get wind variable names for this dataset
    
    # Create output directory within the input directory
    output_dir = os.path.join(input_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Grid resolution: {grid_resolution} m")
    
    # Create UTM grid based on resolution
    utm_x = np.arange(utm_x_min, utm_x_max + grid_resolution, grid_resolution)
    utm_y = np.arange(utm_y_min, utm_y_max + grid_resolution, grid_resolution)
    
    print(f"UTM grid: {len(utm_x)} x {len(utm_y)} points")

    # %% Get list of input files
    input_files = []
    for wy in range(start_wy, end_wy + 1):
        file_name = file_pattern.format(year=wy)
        file_path = os.path.join(input_dir, file_name)
        if os.path.exists(file_path):
            input_files.append((wy, file_path))
        else:
            print(f"  WARNING: File not found: {file_path}")

    print(f"\nFound {len(input_files)} input files to process")
    if len(input_files) == 0:
        print(f"  Skipping {dataset['name']} - no files found!")
        continue

    # %% Read coordinate system info from first valid file (same for all)
    print("\nReading coordinate system from first file...")
    first_file = None
    for wy, fpath in input_files:
        try:
            with nc.Dataset(fpath, 'r') as ncin:
                # Test if file is readable
                _ = ncin.variables['lon']
                first_file = fpath
                break
        except (OSError, KeyError) as e:
            print(f"  WARNING: Cannot read {os.path.basename(fpath)}: {e}")
            continue
    
    if first_file is None:
        print(f"  ERROR: No valid files found for {dataset['name']}. Skipping dataset.")
        continue
    
    print(f"  Using {os.path.basename(first_file)} as reference")

    with nc.Dataset(first_file, 'r') as ncin:
        # Read lat/lon coordinates
        lon_grid = ncin.variables['lon'][:]  # Geographic longitude (2D array)
        lat_grid = ncin.variables['lat'][:]  # Geographic latitude (2D array)
        
        # Read x/y for calculating resolution
        x_native = ncin.variables['x'][:]
        y_native = ncin.variables['y'][:]
        
        # Calculate input resolution
        dx = np.mean(np.diff(x_native))
        dy = np.mean(np.diff(y_native))
        print(f"Input resolution: dx={dx:.1f} m, dy={dy:.1f} m")
        print(f"Input grid: {lat_grid.shape[0]} x {lat_grid.shape[1]} points")
        print(f"Lat range: {lat_grid.min():.4f} to {lat_grid.max():.4f}°N")
        print(f"Lon range: {lon_grid.min():.4f} to {lon_grid.max():.4f}°E")

    print("\n" + "="*70)
    print("Setting up coordinate transformation")
    print("="*70)

    from pyproj import Transformer

    # Create transformer: WGS84 (lat/lon) -> UTM Zone 10N
    transformer_to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
    
    # Convert input lat/lon grid to UTM
    print("Converting input lat/lon to UTM10N...")
    x_input_utm, y_input_utm = transformer_to_utm.transform(lon_grid.flatten(), lat_grid.flatten())
    x_input_utm = x_input_utm.reshape(lon_grid.shape)
    y_input_utm = y_input_utm.reshape(lat_grid.shape)
    
    print(f"Input UTM range:")
    print(f"  x: {x_input_utm.min():.0f} to {x_input_utm.max():.0f} m")
    print(f"  y: {y_input_utm.min():.0f} to {y_input_utm.max():.0f} m")

    # Create output UTM grid
    utm_xx, utm_yy = np.meshgrid(utm_x, utm_y)
    print(f"\nOutput UTM10N grid:")
    print(f"  x: {utm_x.min():.0f} to {utm_x.max():.0f} m ({len(utm_x)} points)")
    print(f"  y: {utm_y.min():.0f} to {utm_y.max():.0f} m ({len(utm_y)} points)")
    print(f"  dx: {np.mean(np.diff(utm_x)):.0f} m")
    print(f"  dy: {np.mean(np.diff(utm_y)):.0f} m")

    # Check if output points are within input domain
    in_bounds = ((utm_xx >= x_input_utm.min()) & (utm_xx <= x_input_utm.max()) &
                 (utm_yy >= y_input_utm.min()) & (utm_yy <= y_input_utm.max()))
    print(f"\nPoints within input domain: {in_bounds.sum()} / {in_bounds.size}")

    print("\n" + "="*70)
    print("Processing files and interpolating")
    print("="*70)

    # Initialize storage for accumulated data
    all_times = []
    all_data = {var_out: [] for var_out in wind_vars.values()}

    # Process each file
    for file_idx, (wy, input_file) in enumerate(input_files):
        print(f"\n[{file_idx+1}/{len(input_files)}] Processing WY{wy}: {os.path.basename(input_file)}")
        
        try:
            with nc.Dataset(input_file, 'r') as ncin:
                # Read time
                time_var = ncin.variables['time']
                time_vals = time_var[:]
                time_units = time_var.units  # e.g., "hours since 1979-10-01" or "hours since 1979-10-01 00:00:00"
                
                print(f"  Time steps: {len(time_vals)}")
                print(f"  Time units: {time_units}")
                
                # Convert time to hours since 1900-01-01
                # Parse the reference time from time_units
                if 'hours since' in time_units or 'hour since' in time_units:
                    ref_str = time_units.split('since')[1].strip()
                    
                    # Try different datetime formats
                    time_origin = None
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%d_%H:%M:%S']:
                        try:
                            time_origin = datetime.strptime(ref_str, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if time_origin is None:
                        raise ValueError(f"Could not parse time origin from: '{ref_str}'")
                else:
                    raise ValueError(f"Unexpected time units: {time_units}")
                
                time_ref_1900 = datetime(1900, 1, 1)
                hours_origin_to_1900 = (time_origin - time_ref_1900).total_seconds() / 3600
                time_hours_1900 = time_vals + hours_origin_to_1900
                all_times.extend(time_hours_1900)
                
                # Check if we need to read terrain height for pressure conversion
                terrain_height = None
                if 'pressure_conversion' in dataset and 'PSFC' in wind_vars:
                    terrain_var = dataset['pressure_conversion']['terrain_height']
                    if terrain_var in ncin.variables:
                        print(f"  Reading terrain height: {terrain_var}")
                        terrain_height = ncin.variables[terrain_var][:]  # May be 2D or 3D
                        # If 3D (time, y, x), take first timestep since terrain is static
                        if terrain_height.ndim == 3:
                            terrain_height = terrain_height[0, :, :]
                        print(f"    Terrain range: {terrain_height.min():.1f} to {terrain_height.max():.1f} m")
                
                # Read and interpolate each wind variable
                for var_in, var_out in wind_vars.items():
                    print(f"  Processing {var_in} -> {var_out}")
                    data_in = ncin.variables[var_in][:]  # shape: (time, y, x)
                    
                    # Convert accumulated radiation to instantaneous flux if needed
                    if var_in in ['ACSWDNB', 'ACLWDNB']:
                        print(f"    Converting accumulated {var_in} to instantaneous flux")
                        print(f"    Accumulated range: {data_in.min():.0f} to {data_in.max():.0f} J m-2")
                        data_in = convert_accumulated_to_instantaneous_flux(data_in, time_hours_1900)
                        print(f"    Instantaneous flux range: {data_in.min():.2f} to {data_in.max():.2f} W m-2")
                        if var_in == 'ACSWDNB':
                            print(f"    Expected: daytime peaks ~600-1100 W m-2, nighttime ~0 W m-2")
                        elif var_in == 'ACLWDNB':
                            print(f"    Expected: relatively constant ~250-450 W m-2")
                    
                    # Convert surface pressure to MSL if needed
                    if (var_in == 'PSFC' and 'pressure_conversion' in dataset and 
                        terrain_height is not None):
                        temp_var = dataset['pressure_conversion']['temperature']
                        if temp_var in ncin.variables:
                            print(f"    Converting surface pressure to MSL pressure")
                            temperature = ncin.variables[temp_var][:]  # shape: (time, y, x)
                            print(f"    Using temperature: {temp_var}")
                            print(f"    Temperature range: {temperature.min():.1f} to {temperature.max():.1f} K")
                            
                            # Convert to MSL
                            data_in = convert_surface_to_msl_pressure(data_in, terrain_height, temperature)
                            
                            print(f"    PSFC range: {ncin.variables[var_in][:].min():.1f} to {ncin.variables[var_in][:].max():.1f} Pa")
                            print(f"    PMSL range: {data_in.min():.1f} to {data_in.max():.1f} Pa")
                    
                    # Prepare output array for this file
                    nt = data_in.shape[0]
                    ny_out = len(utm_y)
                    nx_out = len(utm_x)
                    data_out = np.full((nt, ny_out, nx_out), np.nan, dtype=np.float32)
                    
                    # For regular grid interpolation, we need 1D arrays of x and y coordinates
                    # Extract the centerline coordinates (since grid is regular in native space)
                    x_coords_1d = x_input_utm[0, :]  # x values along first row
                    y_coords_1d = y_input_utm[:, 0]  # y values along first column
                    
                    # Check if grid is sufficiently regular for RegularGridInterpolator
                    x_std = np.std(np.diff(x_coords_1d))
                    y_std = np.std(np.diff(y_coords_1d))
                    
                    if x_std < 0.01 and y_std < 0.01:  # Grid is regular enough
                        print(f"    Using fast RegularGridInterpolator (grid is regular enough)")
                        print(f"    Extrapolation: {'ON' if USE_EXTRAPOLATION else 'OFF'}")
                        
                        # Show progress for regular interpolation too
                        progress_interval = max(1, nt // 20)  # Show ~20 updates
                        for t in range(nt):
                            if t % progress_interval == 0 or t == nt - 1:
                                print(f"    Timestep {t+1}/{nt} ({100*(t+1)/nt:.1f}%)")
                            
                            # Create interpolator for this timestep
                            if USE_EXTRAPOLATION:
                                # Use nearest extrapolation to avoid NaNs
                                interp = RegularGridInterpolator(
                                    (y_coords_1d, x_coords_1d),
                                    data_in[t, :, :],
                                    method='linear',
                                    bounds_error=False,
                                    fill_value=None  # None enables nearest extrapolation
                                )
                            else:
                                # Standard behavior with NaNs outside domain
                                interp = RegularGridInterpolator(
                                    (y_coords_1d, x_coords_1d),
                                    data_in[t, :, :],
                                    method='linear',
                                    bounds_error=False,
                                    fill_value=np.nan
                                )
                            
                            # Interpolate to output grid
                            points = np.column_stack([utm_yy.flatten(), utm_xx.flatten()])
                            data_out[t, :, :] = interp(points).reshape(ny_out, nx_out)
                    
                    else:
                        # Fall back to nearest neighbor for irregular grids (much faster)
                        print(f"    Using NearestNDInterpolator for irregular grid (fast)")
                        print(f"    Extrapolation: always ON (nearest neighbor inherently extrapolates)")
                        from scipy.interpolate import NearestNDInterpolator
                        
                        input_points = np.column_stack([x_input_utm.flatten(), y_input_utm.flatten()])
                        output_points = np.column_stack([utm_xx.flatten(), utm_yy.flatten()])
                        
                        # Build interpolator once - grid doesn't change between timesteps
                        print(f"    Building nearest neighbor tree (one-time setup)...")
                        # Use a dummy array to get the indices/structure
                        dummy_values = np.arange(len(input_points), dtype=np.float32)
                        interp_indices = NearestNDInterpolator(input_points, dummy_values)
                        nearest_indices = interp_indices(output_points).astype(int)
                        
                        # More frequent progress updates
                        progress_interval = max(1, nt // 100)  # Show ~100 updates
                        for t in range(nt):
                            if t % progress_interval == 0 or t == nt - 1:
                                print(f"    Timestep {t+1}/{nt} ({100*(t+1)/nt:.1f}%)")
                            
                            # Simply index into flattened data using pre-computed nearest indices
                            values = data_in[t, :, :].flatten()
                            data_out[t, :, :] = values[nearest_indices].reshape(ny_out, nx_out)
                    
                    # Accumulate data
                    all_data[var_out].append(data_out)
                    print(f"    Completed. Valid: {np.isfinite(data_out).sum()}/{data_out.size}")
        
        except (OSError, KeyError, ValueError, AttributeError) as e:
            print(f"  ERROR processing {os.path.basename(input_file)}: {e}")
            print(f"  Skipping WY{wy}")
            continue

    print("\n" + "="*70)
    print("Concatenating and writing output files")
    print("="*70)

    # Concatenate all years
    all_times = np.array(all_times)
    for var_out in wind_vars.values():
        all_data[var_out] = np.concatenate(all_data[var_out], axis=0)

    print(f"Total time steps: {len(all_times)}")
    print(f"Date range: WY{start_wy} to WY{end_wy}")

    # Write one file per variable
    for var_name in wind_vars.values():
        output_file = os.path.join(output_dir, f'CONUS404_{dataset["name"]}_{var_name}_{start_wy}_{end_wy}_UTM10.nc')
        print(f"\nWriting: {output_file}")
        
        data_out = all_data[var_name]
        print(f"  Shape: {data_out.shape}")
        print(f"  Valid data: {np.isfinite(data_out).sum()}/{data_out.size} ({100*np.isfinite(data_out).sum()/data_out.size:.1f}%)")
        
        with nc.Dataset(output_file, 'w', format='NETCDF4') as ncout:
            # Create dimensions
            ncout.createDimension('time', len(all_times))
            ncout.createDimension('x', len(utm_x))
            ncout.createDimension('y', len(utm_y))
            
            # Create coordinate variables
            time_out = ncout.createVariable('time', 'f8', ('time',), 
                                             chunksizes=(1024,), zlib=False,
                                             fill_value=np.nan)
            time_out[:] = all_times
            time_out.units = 'hours since 1900-01-01'
            time_out.calendar = 'gregorian'
            time_out.long_name = 'time'
            time_out.standard_name = 'time'
            time_out.axis = 'T'
            
            x_out = ncout.createVariable('x', 'f4', ('x',), fill_value=np.nan)
            x_out[:] = utm_x
            x_out.units = 'm'
            x_out.long_name = 'x coordinate (UTM10N)'
            x_out.standard_name = 'projection_x_coordinate'
            x_out.axis = 'X'
            
            y_out = ncout.createVariable('y', 'f4', ('y',), fill_value=np.nan)
            y_out[:] = utm_y
            y_out.units = 'm'
            y_out.long_name = 'y coordinate (UTM10N)'
            y_out.standard_name = 'projection_y_coordinate'
            y_out.axis = 'Y'
            
            # Create data variable
            var_out = ncout.createVariable(var_name, 'f4', ('time', 'y', 'x'),
                                            chunksizes=(1, len(utm_y), len(utm_x)),
                                            zlib=False, fill_value=np.nan)
            var_out[:] = data_out
            var_out.coordinates = 'y x'
            
            # Set attributes based on variable
            if var_name == 'eastward_wind':
                var_out.long_name = '10 metre U wind component'
                var_out.standard_name = 'eastward_wind'
                var_out.units = 'm s**-1'
            elif var_name == 'northward_wind':
                var_out.long_name = '10 metre V wind component'
                var_out.standard_name = 'northward_wind'
                var_out.units = 'm s**-1'
            elif var_name == 'air_temperature':
                var_out.long_name = '2 metre temperature'
                var_out.standard_name = 'air_temperature'
                var_out.units = 'K'
            elif var_name == 'dew_point_temperature':
                var_out.long_name = '2 metre dewpoint temperature'
                var_out.standard_name = 'dew_point_temperature'
                var_out.units = 'K'
            elif var_name == 'air_pressure_fixed_height':
                var_out.long_name = 'Mean sea level pressure'
                var_out.standard_name = 'air_pressure_at_mean_sea_level'
                var_out.units = 'Pa'
            elif var_name == 'surface_solar_radiation':
                var_out.long_name = 'Surface solar radiation'
                var_out.standard_name = 'surface_downwelling_shortwave_flux_in_air'
                var_out.units = 'W m**-2'
            elif var_name == 'surface_thermal_radiation_downwards':
                var_out.long_name = 'Surface thermal radiation downwards'
                var_out.standard_name = 'surface_downwelling_longwave_flux_in_air'
                var_out.units = 'W m**-2'
            
            # Global attributes
            ncout.title = f'CONUS404 {dataset["description"]} wind data interpolated to UTM Zone 10N (WY{start_wy}-{end_wy})'
            ncout.source = f'Interpolated from {dataset["name"]} files'
            ncout.institution = 'USGS CONUS404'
            ncout.history = f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            ncout.Conventions = 'CF-1.8'
            ncout.input_resolution_m = f'{dx:.1f}'
            ncout.output_resolution_m = f'{grid_resolution}'
            ncout.output_projection = 'UTM Zone 10N (EPSG:32610)'
            ncout.interpolation_method = 'linear'
            ncout.extrapolation = 'nearest' if USE_EXTRAPOLATION else 'none'
            ncout.water_years = f'{start_wy}-{end_wy}'
            ncout.n_files_processed = len(input_files)
            ncout.dataset_name = dataset['name']
        
        print(f"  Completed!")

    print(f"\n{dataset['name']} processing complete!")
    print(f"Output files in: {output_dir}")

# %% Final summary
print("\n" + "="*70)
print("ALL DATASETS PROCESSED!")
print("="*70)

