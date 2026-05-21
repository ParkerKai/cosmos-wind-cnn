"""
Download data

Usage:


Output:


"""

# ------------------------------------------------------------------------------
# Import packages
# ------------------------------------------------------------------------------
import os
from cosmos_wind_cnn.data.logging_config import get_logger


# ------------------------------------------------------------------------------
# Startup and general housecleaning
# ------------------------------------------------------------------------------

logger = get_logger("DownloadLogger", log_file="..\case_studies\puget_soud\logs\DataDownload.log")

# ------------------------------------------------------------------------------
# Get it done
# ------------------------------------------------------------------------------


def main():
    # ------------------------------------------------------------------------------
    # User Inputs
    # ------------------------------------------------------------------------------

    # Output directory
    dir_out = r"..\case_studies\puget_sound\data\raw"

    # Spatial boundaries
    lim_lon = [-126, -121.5]  # West, East
    lim_lat = [46.5, 49.5]  # South, North

    # Variables

    # Threads
    Threads = os.cpu_count() - 1

    # Which dataset to download:
    dataset_to_download = "conus404"  # or "conus404"

    # ------------------------------------------------------------------------------
    # Download the ERA5 Data
    # ------------------------------------------------------------------------------
    if dataset_to_download == "era5":
        from cosmos_wind_cnn.data.era5_download import download_era5

        logger.info("Starting ERA5 download...")

        download_era5(
            output_dir=os.path.join(dir_out, "ERA5"),
            area=[
                lim_lat[1],
                lim_lon[0],
                lim_lat[0],
                lim_lon[1],
            ],  # North, West, South, East
            variables=[
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "2m_dewpoint_temperature",
                "mean_sea_level_pressure",
                "total_precipitation",
            ],
            dataset="reanalysis-era5-land",
            max_threads=Threads,
        )

    # ------------------------------------------------------------------------------
    # Download the Conus404 Data
    # ------------------------------------------------------------------------------
    elif dataset_to_download == "conus404":
        from cosmos_wind_cnn.data.conus404 import conus404_download_subset

        logger.info("Starting conus404 download...")

        # Dataset selection
        dataset = "conus404-hourly-osn"

        # Variables to download (BA dataset will drop variables since it has a fixed variable list)
        var = ["T2", "TD2", "U10", "V10", "PSFC"]

        # Call the function
        conus404_download_subset(
            output_dir=os.path.join(dir_out, "conus404"),
            lim_lon=lim_lon,
            lim_lat=lim_lat,
            dataset=dataset,
            variables=var,
            reproject=False,
            n_workers=Threads,
        )

    # ------------------------------------------------------------------------------
    # Write out the yaml file with the metadata for this download
    # ------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
