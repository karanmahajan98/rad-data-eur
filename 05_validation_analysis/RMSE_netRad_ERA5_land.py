# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 15:22:36 2025

@author: Karan Mahajan

Calculate RMSE between ERA5-Land and NetRad-GEB datasets
"""

import xarray as xr
import os
from glob import glob
import dask
from time import time
import xarray_regrid
import numpy as np

#Load the two datasets
era5_files = glob(os.path.join('data/ERA5_land_rad_daily', '*.nc'))
net_rad_files = glob(os.path.join('data/net_rad_emo_era', '*.nc'))

print('Loading datasets')
era5 = xr.open_mfdataset(era5_files, chunks = {} , parallel = True)
netrad = xr.open_mfdataset(net_rad_files, chunks = {'time': 365, 'lon': 380, 'lat':495 },engine='h5netcdf', parallel = True)
print('Datasets loaded\n')

#Regrid to ERA5-Land resolution
netrad_regrided = netrad.regrid.conservative(era5, nan_threshold = 0.5)

# RMSE calculation

t0 = time()
# Extract the relevant variables
era5_var = era5['snsr']
rad_var = netrad_regrided['net_radiation']

#  Calculate the difference
difference = era5_var - rad_var

# Square the difference
squared_difference = difference**2

#Calculate the mean along the 'time' dimension
mean_squared_difference = squared_difference.mean(dim='time')

# Take the square root to get the RMSE
rmse = np.sqrt(mean_squared_difference)

# Name the resulting DataArray
rmse.name = 'RMSE'

print(f'Time taken to calcualate RMSE lazily: {(time()-t0)/60} minutes\n')

rmse_ds = xr.Dataset({
    'rmse': rmse
})

# Save as netCDF
t0 = time()
print('Saving RMSE as netCDF file')
rmse_ds.to_netcdf('data/rmse_netRad_emo_era.nc', engine='h5netcdf')
print(f'Time taken to save file: {(time()-t0)/60} minutes')



# Now also calculate seasonal RMSE
seasonal_mse = squared_difference.groupby('time.season').mean(dim='time')

seasonal_rmse = np.sqrt(seasonal_mse)

seasonal_rmse.name = 'seasonal_RMSE'

seasonal_rmse_ds = xr.Dataset({
    'seasonal_rmse': seasonal_rmse
})

t0 = time()
print('Saving Seasonal RMSE as netCDF file')
seasonal_rmse_ds.to_netcdf('data/seasonal_rmse_netRad_emo_era.nc', engine='h5netcdf')
print(f'Time taken to save file: {(time()-t0)/60} minutes')