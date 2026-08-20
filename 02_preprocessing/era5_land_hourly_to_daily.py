# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 11:27:44 2025

@author: Karan Mahajan

convert ERA5-Land-Hourly radiation data to daily scale
Clip to desired date range
Save as yearly netCDF files
"""
import xarray as xr
import pandas as pd
import os
from glob import glob
import dask
from time import time

# Find all the files
era5_files = glob(os.path.join('data/era5_land_hourly_rad_eur/**', '*.nc'))

#Load dataset
print('Loading the datasets\n')
ERA5_ds = xr.open_mfdataset(era5_files, parallel=True)
print('Loading completed\n')


# Select the desired time range
ERA5_ds_clip = ERA5_ds.sel(valid_time=slice('2001-01-02T00:00', '2021-01-01T00:00')) # Note the data structure: shifted by one day

# Resample to daily data and use time delta
daily_era5 = ERA5_ds_clip.sel(valid_time=(ERA5_ds_clip.valid_time.dt.hour == 0)) # for accumulated vars, take the value at 0 hour of next day (ERA5_Land documentation)
daily_era5["valid_time"] = daily_era5["valid_time"] - pd.Timedelta(days=1)

# Convert the units to W/m2
daily_era5['ssr'] = ((daily_era5['ssr'])/86400).astype('float32')  # By default python converts to float 64 which doubles the array size
daily_era5['str'] = ((daily_era5['str'])/86400).astype('float32')

#Also calculate net radiation
daily_era5['snsr'] = daily_era5['ssr'] + daily_era5['str']

# Change the name of the coordinates for consistency across datasets
daily_era5 = daily_era5.rename({'longitude': 'lon', 'latitude' : 'lat', 'valid_time' : 'time'})
daily_era5 = daily_era5.drop_vars(['number', 'expver']) # Not needed

# Save as netCDF files
t_tot_0 = time()
for year, data in daily_era5.groupby('time.year'):
    t0 = time()
    print(f'Currently saving netCDF file for year : {year}')
    data.to_netcdf(f'data/ERA5_land_rad_daily_{year}.nc', engine='h5netcdf')
    print(f"Time elapsed to save netCDF file for year {year}: {(time()-t0)/60} minutes\n")
    
print(f'Time elapsed to save all the netCDF files: {(time()-t_tot_0)/60} minutes')