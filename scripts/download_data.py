"""
Dwnload data

Usage:


Output:


"""


### Import functions ###

from cosmos_wind_cnn.data.download import download_era5, download_conus404_subset


def main():
    """
    Script to download data

    """

    # Output directory
    dir_out = r"D:\Kai\DataDownloads\Conus404\download_BA_SFbay"

    # Spatial boundaries
    lim_lon = [-123.6, -120.7]  # West, East
    lim_lat = [36.7, 39.4]  # South, North

    asdf
    # ------------------------------------------------------------------------------
    # Download the ERA5 Data
    # ------------------------------------------------------------------------------

    download_era5(
        dir_out=r"D:\Kai\ERA5\PNW_Meteo",
        area_lims=[49, -126.5, 41.5, -122],
        years=[2024, 2026],
        months=[1, 12],
        variables=[
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "2m_dewpoint_temperature",
            "mean_sea_level_pressure",
            "total_precipitation",
        ],
        dataset="reanalysis-era5-land",
        max_threads=10,
    )

    # ------------------------------------------------------------------------------
    # Download the Conus404 Data
    # ------------------------------------------------------------------------------

    # Dataset selection
    dataset = "conus404-hourly-ba-osn"

    # Optional subset of variables (None = defaults)
    var = None

    # Temporal subset (None = full 1979–2022)
    t_lims = None

    # Whether to reproject and EPSG code for target CRS
    reproject = False
    reproject_crs = 26910  # UTM Zone 10N

    # Number of Dask workers
    n_workers = 2

    # Call the function
    download_conus404_subset(
        dir_out=dir_out,
        lim_lon=lim_lon,
        lim_lat=lim_lat,
        dataset=dataset,
        var=var,
        t_lims=t_lims,
        reproject=reproject,
        reproject_crs=reproject_crs,
        n_workers=n_workers,
    )

    # ------------------------------------------------------------------------------
    # Write out the yaml file with the metadata for this download
    # ------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
