# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 17:39:43 2025

@author: Karan Mahajan

This script:
1. combines all the PAR and PARdiff files into a single xarray dataset
2. Regrids this dataset to the GLASS albedo dimensions
3. Clips the dataset to European scale (Extent of EMO1 data)
4. Saves yearly netCDF files

Update: There is no need to process GLASS albedo here (forward fill) because I already saved a process GLASS albedo product using another script.
"""

import xarray as xr
import pandas as pd
import os
from glob import glob
import re
import matplotlib.pyplot as plt
import xarray_regrid
from time import time

# Function to open and tag each file with correct time
def open_with_time(filepath):
    """
    Opens a single day netCDF file for PAR or PARDiff and adds the time information

    Parameters
    ----------
    filepath : string
        path for a single nc file.

    Returns
    -------
    ds
        xarray dataset.
    """    
    # Extract date from filename (e.g. A2001252)
    match = re.search(r"A(\d{4})(\d{3})", os.path.basename(filepath))
    if not match:
        raise ValueError(f"Could not extract date from {filepath}")
    year, doy = int(match[1]), int(match[2])
    if (year < 2001 or year > 2020) :                               # Only consider relevant years
        return
    time = pd.to_datetime(f"{year}-{doy:03d}", format="%Y-%j")
    
    # Open file
    ds = xr.open_dataset(filepath, chunks={})
    
    # Add time dimension
    return ds.expand_dims({"time": [time]})


# Open the clipped GLASS albedo (only clipped for space, not time)
glass_alb_clipped = xr.open_dataset("data/GLASS_albedo_combined.nc",chunks = {'time': 200, 'lat': 900, 'lon': 900})
glass_alb_clipped = glass_alb_clipped.sel(time=slice("2001-01-01", "2020-12-31"))

# Fill in the missing dates in the GLASS albedo dataset
new_time = pd.date_range(
    start=glass_alb_clipped.time.min().item(), 
    end="2020-12-31", 
    freq="D"
)

BWalbedo_clipped = glass_alb_clipped.reindex(time=new_time, method="ffill").transpose('time','lat','lon')


PAR_folder = "data/BESS_PAR_all"
PAR_files = sorted(glob(os.path.join(PAR_folder, "BESS_PAR_Daily.A*.nc")))

# Load all datasets and concatenate
t0 = time()
datasets = [open_with_time(f) for f in PAR_files]
datasets_filtered = [ds for ds in datasets if ds is not None]
PAR_combined = xr.concat(datasets_filtered, dim="time").sortby("time")
print(f"Time elapsed to load all the PAR files: {(time()-t0)/60} minutes")

PARdiff_folder = "data/BESS_PAR_diff_all"
PARdiff_files = sorted(glob(os.path.join(PARdiff_folder, "BESS_PARDiff_Daily.A*.nc")))

# Load all datasets and concatenate
t0 = time()
PARdiff_datasets = [open_with_time(f) for f in PARdiff_files]
PARdiff_datasets_filtered = [ds for ds in PARdiff_datasets if ds is not None]
PARdiff_combined = xr.concat(PARdiff_datasets_filtered, dim="time").sortby("time")
print(f"Time elapsed to load all the PAR diff files: {(time()-t0)/60} minutes")

# Extent of the grid
lon_min = -25.241666666666667
lon_max = 50.241666666666660
lat_min = 22.7583333333333328
lat_max = 72.241666666666674

# Assuming 0.05° grid resolution (common for global datasets)
dx = 0.05  # longitude resolution
dy = 0.05  # latitude resolution

# Expand bounds by half-cell to capture edge-touching cells
buffer_lon = dx / 2
buffer_lat = dy / 2

# keeping some buffer distance to the extreme ends of the grid
lon_min_exp = lon_min - buffer_lon
lon_max_exp = lon_max + buffer_lon
lat_min_exp = lat_min - buffer_lat
lat_max_exp = lat_max + buffer_lat

# CLIP dataset extents
PAR_clipped = PAR_combined.sel(
    lon=slice(lon_min_exp, lon_max_exp),
    lat=slice(lat_max_exp, lat_min_exp)  # Reverse order for latitude (usually N to S)
)

PARdiff_clipped = PARdiff_combined.sel(
    lon=slice(lon_min_exp, lon_max_exp),
    lat=slice(lat_max_exp, lat_min_exp)  
)

# Data for one date is missing in PARDiff product. Copy the previous date's data

missing_date = pd.Timestamp("2001-06-25")      # This date is known from previous analysis
previous_date = pd.Timestamp("2001-06-24") 
prev_data = PARdiff_clipped.sel(time=previous_date)
new_data = prev_data.expand_dims(time=[missing_date])
PARdiff_clipped_new = xr.concat([PARdiff_clipped, new_data], dim="time").sortby("time")

# Regrid to match the GLASS albedo grid
PAR_clipped_regrided = PAR_clipped.regrid.nearest(BWalbedo_clipped)
PARdiff_clipped_regrided = PARdiff_clipped_new.regrid.nearest(BWalbedo_clipped)

# Save as netCDF files
t0 = time()
for year, data in PAR_clipped_regrided.groupby('time.year'):
    data.to_netcdf(f'data/PAR_clipped_regrid_GLASS_{year}.nc', engine='h5netcdf')

print(f"Time elapsed to save all PAR to netCDF files: {(time()-t0)/60} minutes")


# Save as netCDF files
t0 = time()
for year, data in PARdiff_clipped_regrided.groupby('time.year'):
    data.to_netcdf(f'data/PARDiff_clipped_regrid_GLASS_{year}.nc', engine='h5netcdf')

print(f"Time elapsed to save all PARDiff to netCDF files: {(time()-t0)/60} minutes")