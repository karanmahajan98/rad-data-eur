# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 09:56:52 2025

@author: Karan Mahajan

This script:
    1. Combines all the GLASS albedo hdf files 
    2. Clips spatial extent to Europe (extents from EMO-1)
    3. Fills the time intervals at daily resolution using forward fill
    4. Saves yearly netCDF files 
"""

import xarray as xr
import rioxarray as rxr
import pandas as pd
import os
from glob import glob
import dask
from time import time

def open_albedo_pair(hdf_path, subdatasets):
    """
    Takes a GLASS albedo hdf file and converts it into an xarray dataset with 
    date and the necessary variables

    Parameters
    ----------
    hdf_path : string
        Path of the hdf file to process.
    subdatasets : dict
        Dict with key as the variable name and the value as the HDF code for that variable.
        Extracted by checking a single hdf file metadata

    Returns
    -------
    xarray dataset
        Converted xarray dataset.

    """
    data_vars = {}
    # Extract date from filename (e.g., A2001089 → 2001-03-30)
    date = pd.to_datetime(os.path.basename(hdf_path).split(".")[2][1:], format="%Y%j")

    for key, sds_fmt in subdatasets.items():
        gdal_path = sds_fmt.format(hdf_path)
        da = rxr.open_rasterio(gdal_path, masked=True, chunks={"x": 512, "y": 512})
        da = da.squeeze("band", drop=True)
        da = da.assign_coords(time=date)
        data_vars[key] = da

    # Combine the variables into a Dataset
    return xr.Dataset(data_vars)


dask.config.set({"array.chunk-size": "100MB"})

# Path to the GLASS albedo data
base = "data/GLASS_albedo"
hdf_files = sorted(glob(os.path.join(base, "*", "*.hdf")))


# These are the subdataset keys we want (update based on actual names if needed)
subdatasets = {
    "BSA_shortwave": 'HDF4_EOS:EOS_GRID:"{}":GLASS02B03:BSA_shortwave',  # black sky albedo
    "WSA_shortwave": 'HDF4_EOS:EOS_GRID:"{}":GLASS02B03:WSA_shortwave',  # white sky albedo
    "QC": 'HDF4_EOS:EOS_GRID:"{}":GLASS02B03:QC'                         # quality flag
}


# Load and Concatenate 
print('Combining all the .hdf files to xarray dataset')
t0 = time()
datasets = [open_albedo_pair(f, subdatasets) for f in hdf_files]
combined_ds = xr.concat(datasets, dim="time").sortby("time")
combined_ds = combined_ds.rename({"x": "lon", "y": "lat"})
print(f"Time elapsed to load and combine all hdf files: {(time()-t0)/60} minutes\n")


# Now clip to Europe extent (Extracted from EMO-1 data)
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

# Expanded selection bounds
lon_min_exp = lon_min - buffer_lon
lon_max_exp = lon_max + buffer_lon
lat_min_exp = lat_min - buffer_lat
lat_max_exp = lat_max + buffer_lat

# Find nearest grid cells beyond expanded bounds
clipped_ds = combined_ds.sel(
    lon=slice(lon_min_exp, lon_max_exp),
    lat=slice(lat_max_exp, lat_min_exp)  # Reverse order for latitude (usually N→S)
)

print(clipped_ds)
print(" ")

# Fill in the missing dates in the clipped GLASS albedo dataset (from 8 day to daily frequency)
new_time = pd.date_range(
    start=clipped_ds.time.min().item(), 
    end="2020-12-31", 
    freq="D"
)

BWalbedo_clipped = clipped_ds.reindex(time=new_time, method="ffill").transpose('time','lat','lon')
print(BWalbedo_clipped)
print(" ")


# Save as netCDF (one file for every year)
t0 = time()
for year, data in BWalbedo_clipped.groupby('time.year'):
    print(f"Currently saving netcdf file for year: {year}")
    data.to_netcdf(f'data/GLASS_alb_combined_clipped_ffill_{year}.nc', engine='h5netcdf')
    print()
    print(f"Time taken for year {year} : {(time()-t0)/60} minutes\n")
print(f"Time elapsed to save all files as netCDF: {(time()-t0)/60} minutes")








