#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:07:13 2026

@author: Karan Mahajan
"""

import xarray as xr
import numpy as np
import glob
import os
from time import time

t0 = time()

input_folder = '/data/net_rad_eur'
output_folder = '/data/net_rad_eur_CF'

file_paths = sorted(glob.glob(os.path.join(input_folder, 'net_rad_eur_*.nc')))

print(f"Found {len(file_paths)} files. Starting processing...")

for file_path in file_paths:
    
    t1 = time()
    
    file_name = os.path.basename(file_path)
    
    if file_name == 'net_rad_eur_2001.nc':
        continue
    
    print(f"Processing {file_name}...\n")
    
    ds = xr.open_dataset(file_path, chunks = {})
    
    del ds.attrs['spatial_res'], ds.attrs['temporal_res']
    
    ds.attrs['Conventions'] = 'CF-1.11'
    ds.attrs['title'] = 'Surface Net Radiation for Europe and North Africa'
    ds.attrs['institution'] = 'Helmholtz Centre for Environmental Research, Permoserstrasse 15, Leipzig, Germany'
    ds.attrs['author'] = 'Karan Mahajan (UFZ), Ye Tuo (TUM), and Jian Peng (UFZ). Contact karan.mahajan@ufz.de'
    ds.attrs['source'] = 'Calculated using GLASS albedo, BESS PAR, BESS PARDiff, EMO1 (v3.0.0), ELITE SLUR, and ELITE SLDR.'
    ds.attrs['history'] = '23 june 2026 added CF metadata'
    ds.attrs['spatial_resolution'] = '1 arcmin'
    ds.attrs['temporal_resolution'] = 'daily'
    
    ds['crs'] = xr.DataArray(np.int32(1))
    ds['crs'].attrs = {
            'grid_mapping_name': 'latitude_longitude'
        }
    
    ds['lat'].attrs = {
                'standard_name': 'latitude',
                'long_name': 'latitude',
                'units': 'degrees_north',
                'axis': 'Y'}
    
    ds['lon'].attrs = {
                'standard_name': 'longitude',
                'long_name': 'longitude',
                'units': 'degrees_east',
                'axis': 'X'
            }   
    
    ds['time'].attrs = {
                'standard_name': 'time',
                'long_name': 'time',
                'axis': 'T'
            }
    
    ds['net_radiation'].attrs = {
                'standard_name': 'surface_net_radiative_flux',
                'long_name': 'Net radiation at the surface',
                'units': 'W m-2',
                'grid_mapping': 'crs'
            }
    
    
    output_path = os.path.join(output_folder, file_name)
    
    encoding = {
        'net_radiation': { 'zlib': True, 'complevel': 5}
        }
    
    print(f"Saving {file_name} to netCDF")
    
    ds.to_netcdf(output_path, engine='h5netcdf', encoding=encoding)
    
    ds.close()
    
    del ds
    
    print(f'Successfully saved {file_name} with total time= {(time()-t1)/60} minutes')
    
print(f'Total time to save all netCDF files = {(time()-t0)/60} minutes')

