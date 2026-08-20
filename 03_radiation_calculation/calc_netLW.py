# -*- coding: utf-8 -*-
"""
Read TIF files created from ELITE and save them as Zarr.

This script only processes 5 years of data to parallelise the processing a bit.
But this can easily be adapted to process the entire 20 year range.

@author: Karan Mahajan
"""


import re
import glob
import pandas as pd
import xarray as xr
import rioxarray as rxr
from time import time
import xarray_regrid


def open_year(folder, tag, year):           
    """
    Opens all the tif files for a single year as a single dataset

    Parameters
    ----------
    folder : str
        Path for tif files folder.
    tag : str
        variable to process: SLUR or SLDR.
    year: int
        year to process

    Returns
    -------
    ds: xarray dataset
        combined dataset for one varible for one year.

    """
    files  = sorted(glob.glob(f"{folder}/{tag}_*.tif"))

    # Build the time vector once 
    times = pd.to_datetime(
        [re.search(r"(\d{4})(\d{3})", f).group()      # e.g. Julian date "2001117"
         for f in files],
        format="%Y%j"
    )
    
    assert times.is_monotonic_increasing, "File order doesn't match chronological order!"

    # xr.open_mfdataset + engine="rasterio" is the fast way to get a lazy stack
    ds = xr.open_mfdataset(
        files,
        engine="rasterio",
        concat_dim="time",
        combine="nested",
        chunks={"band":1, "y":2048, "x":2048},              # tune this once
        parallel=False,
        preprocess=lambda da: da.squeeze("band", drop=True)
                              .rename({"y":"lat", "x":"lon"})
    )
    # Add time information
    ds = ds.assign_coords(time=("time", times))
    
    # Time range to be used to fill temporal gaps using forward fill
    new_time = pd.date_range(
        start= f"{year}-01-01", 
        end= f"{year}-12-31", 
        freq="D"
    )

    ds = ds.reindex(time=new_time, method="ffill")
    
    return ds

t_tot = time()

# open emo data for regridding
emo = xr.open_dataset('/data/twd_data/ANN_PT/raw_datasets/EMO1_v3/EMO1_shrt_rad/rg_2002.nc', chunks = {})
emo_grid = emo.isel(time=0, drop=True)

# specify paths
root_slur = "/data/twd_data/ANN_PT/raw_datasets/ELITE/ELITE_SLUR_tifs"
root_sldr = "/data/twd_data/ANN_PT/raw_datasets/ELITE/ELITE_SLDR_tifs"
out_dir = "/work/mahajan/net_LW"

for year in range(2010,2011):
    
    t0_year = time()
    print(f"Currently processing year : {year}")

    print("   Opening all SLUR tifs for the given year")
    ds_slur = open_year(f"{root_slur}/{year}", "SLUR", year)
    print("   Opening all SLDR tifs for the given year")
    ds_sldr = open_year(f"{root_sldr}/{year}", "SLDR", year)
    
    net_lw  = (ds_sldr - ds_slur)
    
    print("   Regridding")
    lw_regrid = net_lw.regrid.conservative(emo_grid, nan_threshold = 0.5)
    lw_regrid = lw_regrid.rename({"band_data":"net_longwave"})
    lw_regrid = lw_regrid.drop_vars(["spatial_ref"], errors="ignore")
    print("   Completed regridding")
    
    lw_regrid = lw_regrid.chunk({
                "time": 90,    # 3 months per chunk
                "lat": 1000,
                "lon": 1000
                })

    # save as zarr
    print("   Saving as zarr")
    zarr_path = f"{out_dir}/netLW_{year}.zarr"
    lw_regrid.to_zarr(zarr_path, mode="w", consolidated=True)

    print(f"Completed processing year {year}. Time taken: {(time()-t0_year)/60} minutes\n")
    
    del ds_slur, ds_sldr, net_lw, lw_regrid 
    
    
print(f"Processed all years. Time taken: {(time()-t_tot)/3600} hours")


