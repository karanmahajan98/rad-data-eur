# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:45:06 2025

@author: Karan Mahajan

This script:
    1. Loads all the GLASS albedo preprocessed files
    2. Loads the PAR and PARDiff preprocessed files
    3. Calculates diffuse fraction and then calculates blue sky albedo
    4. Saves yearly netCDF files for blue sky albedo
"""

import xarray as xr
import os
from glob import glob
import dask
from time import time

# Specify file locations
PAR_list = glob(os.path.join('data/PAR_clipped_regrid_GLASS', '*.nc'))
PARDiff_list = glob(os.path.join('data/PARDiff_clipped_regrid_GLASS','*.nc'))
glass_alb_list = glob(os.path.join('data/GLASS_alb_combined_clipped_ffill', '*.nc'))
del glass_alb_list[0] # don't need data for year 2000

# Load datasets
t0 = time()
PAR_ds = xr.open_mfdataset(PAR_list, chunks = {'time': 365, 'lat': 330, 'lon':  302}, parallel = True)
PARdiff_ds = xr.open_mfdataset(PARDiff_list, chunks = {'time': 365, 'lat': 330, 'lon':  302}, parallel = True)
glass_alb_ds = xr.open_mfdataset(glass_alb_list, chunks = {'time': 365, 'lat': 330, 'lon':  302}, parallel = True)
print(f'Time elapsed to load all netCDf files: {(time()-t0)/60} minutes')

#Calculate diffuse fraction in PAR

PARDiff = PARdiff_ds['PARDiff']
PAR     = PAR_ds['PAR']
f_dif_raw = xr.where(PAR == 0, 0, PARDiff / PAR) # If PAR is 0, let the diffuse fraction also be 0, else the ratio

# Cap values >1 to 1 (inherent error in the data?)
f_dif_capped = xr.where(f_dif_raw > 1, 1, f_dif_raw)

# Caculate blue sky albedo
# Blue sky albedo = (direct_fraction*black_sky_albedo)  +   (diffuse_fraction*white_sky_albedo)

bsa = glass_alb_ds['BSA_shortwave']
wsa = glass_alb_ds['WSA_shortwave']
f_dir = (1 - f_dif_capped) # Direct fraction
alb_blue = (f_dir * bsa) + (f_dif_capped * wsa)
blue_albedo_ds = (xr.Dataset({'alb_blue': alb_blue}))

t_tot_0 = time()
for year, data in blue_albedo_ds.groupby('time.year'):
    t0 = time()
    print(f'Currently saving netCDF file for year : {year}')
    data.to_netcdf(f'data/blue_sky_albedo_eur_5km_{year}.nc', engine='h5netcdf')
    print(f"Time elapsed to save netCDF file for year {year}: {(time()-t0)/60} minutes\n")
    
print(f'Time elapsed to save all the netCDF files: {(time()-t_tot_0)/60} minutes')
