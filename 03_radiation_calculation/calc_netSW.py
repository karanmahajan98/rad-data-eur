# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:54:09 2025

@author: Karan Mahajan

This script:
    1. Loads the EMO1 shortwave radiation v3 and the prepared blue sky albedo files
    2. Regrid blue sky albedo to EMO1 resolution
    3. Calculates net shortwave radiation
"""

import xarray as xr
import pandas as pd
import os
from glob import glob
import dask
from time import time
import xarray_regrid

#Paths for the netCDf files
EMO1_paths = glob(os.path.join(r"data/EMO1_shrt_rad",'*.nc'))
del EMO1_paths[0]    # Removing years 2000, 2021
del EMO1_paths[-1]

alb_paths = glob(os.path.join(r'data/blue_sky_albedo_eur_5km','*.nc'))

# Open the datasets
# EMO1 comes with pre defined spatial chunks (from the warning when specifying chunks) so only specifying temporal chunks here
print('Loading all the input netCDF files\n')

EMO1_rad = xr.open_mfdataset(EMO1_paths, chunks = {'time':30}, parallel=True)
blue_alb = xr.open_mfdataset(alb_paths, chunks = {'time': 365, 'lat': 330, 'lon':  302}, parallel=True)

# Correct the time dims for EMO1: time is displaced by one day because time steps mark the end of a day (beginning of the next day at 00 hr)
EMO1_corrected = EMO1_rad.assign_coords(time=EMO1_rad.time - pd.Timedelta(days=1))

# Regrid blue albedo to EMO1 spatial resolution 
# Note: EMO1_rad is aroung 1.4km and blue_alb is around 5km resolution
blue_alb_regrided = blue_alb.regrid.nearest(EMO1_corrected) # Note that this generates a very big chunk (automatically breaks down later into smaller chunks)

print(EMO1_corrected)
print(' ')
print(blue_alb_regrided)


# Calculate  net shortwave radiation
R_down = EMO1_corrected['rg'] /86400        # daily downwelling radiation: convert J/m2 to W/m2
alpha_blue = blue_alb_regrided['alb_blue']  # blue-sky albedo

# Compute upwelling and net radiation
R_up = alpha_blue * R_down
R_net = R_down - R_up

# Wrap in dataset
rad_fluxes = xr.Dataset({
    'shortwave_net': R_net
})

print(' ')
print(rad_fluxes)
print(' ')

# Save as netCDF files

# Specify compression encoding
encoding = {
    'shortwave_net': { 'zlib': True, 'complevel': 5}
    }


t_tot_0 = time()
for year, data in rad_fluxes.groupby('time.year'):
    t0 = time()
    print(f'Currently saving netCDF file for year : {year}')
    data.to_netcdf(f'data/net_shortwave_eur_{year}.nc', engine='h5netcdf', encoding=encoding)
    print(f"Time elapsed to save netCDF file for year {year}: {(time()-t0)/60} minutes\n")
    
print(f'Time elapsed to save all the netCDF files: {(time()-t_tot_0)/60} minutes')