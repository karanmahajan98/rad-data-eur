#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karan mahajan

Build per-site time series dataframes (obs vs EMOELITE vs ERA5-Land) for
visual inspection of sites with bad scores. No scoring here, just the
aligned time series.
"""
import pandas as pd
import numpy as np
import os
import glob
import xarray as xr
import pickle

#%% load the fluxnet site data that was processed in another script (process_fluxnet.py)
with open('/data/saved_vars/site_data', 'rb') as file:
    site_data = pickle.load(file)

# one entry per product we want to compare against FLUXNET
dataset_configs = [
    {"dsName": "emoElite", "folderPath": "/data/net_rad",
     "varName": "net_radiation",
     "dimNames": {"time": "time", "lat": "lat", "lon": "lon"}},
    {"dsName": "era5Land", "folderPath": "/data/era5_land_rad",
     "varName": "snsr",
     "dimNames": {"time": "time", "lat": "lat", "lon": "lon"}},
]

#%% 
def open_ds(dsName, folderPath, varName, dimNames):
    """
    Opens a netCDF dataset using xarray
    """
    nc_files = sorted(glob.glob(os.path.join(folderPath, "*.nc")))

    assert len(nc_files) > 0, f"no .nc files found in {folderPath}"
    print(f"Found {len(nc_files)} netCDF files for {dsName}")
    print(nc_files)

    ds = xr.open_mfdataset(nc_files, chunks={})

    assert varName in ds, f"variable '{varName}' not in dataset"
    for dim in [dimNames["time"], dimNames["lat"], dimNames["lon"]]:
        assert dim in ds[varName].dims, f"dimension '{dim}' missing"

    grid_time = pd.to_datetime(ds[dimNames["time"]].values).normalize()
    lon_min, lon_max = float(ds[dimNames["lon"]].min()), float(ds[dimNames["lon"]].max())
    print(ds[varName].dims, ds[varName].shape)
    print(f"lon range: {lon_min} to {lon_max}")

    return ds, grid_time, lon_max

#%% extract the nearest-cell time series for a site, no scoring involved

def extract_series(dsName, ds, grid_time, lon_max, varName, dimNames, site_data):
    """
    loop over sites for a single dataset and pull out the nearest grid cell
    time series, indexed by date. No pairing with obs or scoring here -
    that happens later once both datasets are extracted.

    Returns
    -------
    series_by_site: dict {site_id: pd.Series} of grid values, indexed by date
    skipped: dataframe of site_id + reason (lat/lon problems only, at this stage)
    """
    series_by_site = {}
    skipped = []

    for site_id, df in site_data.items():
        print(f'Currently processing site: {site_id}')
        try:
            lat = float(df["LOCATION_LAT"].iloc[0])
            lon = float(df["LOCATION_LONG"].iloc[0])
        except (ValueError, TypeError):
            skipped.append({"site_id": site_id, "reason": f"[{dsName}] lat/lon missing or not numeric"})
            continue

        if not (np.isfinite(lat) and np.isfinite(lon)):
            skipped.append({"site_id": site_id, "reason": f"[{dsName}] lat/lon not finite"})
            continue

        # if the grid uses 0-360 longitudes but the site is negative, shift it
        lon_q = lon
        if lon_max > 180 and lon < 0:
            lon_q = lon + 360

        # find the nearest cell in the dataset to the site
        cell = ds[varName].sel(**{dimNames["lat"]: lat, dimNames["lon"]: lon_q}, method="nearest")
        grid = pd.Series(cell.values.astype(float), index=grid_time, name=dsName)

        series_by_site[site_id] = grid

    return series_by_site, pd.DataFrame(skipped)

#%% open both datasets

opened_datasets = {}
for cfg in dataset_configs:
    ds, grid_time, lon_max = open_ds(cfg["dsName"], cfg["folderPath"], cfg["varName"], cfg["dimNames"])
    opened_datasets[cfg["dsName"]] = {"ds": ds, "grid_time": grid_time, "lon_max": lon_max}

#%% run extract_series() once per product

series_by_dataset = {}
skipped_all = []

for cfg in dataset_configs:
    dsName = cfg["dsName"]
    print(f'..........Processing dataset: {dsName}.........\n')
    opened = opened_datasets[dsName]
    series_by_site, skipped_df = extract_series(
        dsName, opened["ds"], opened["grid_time"], opened["lon_max"],
        cfg["varName"], cfg["dimNames"], site_data,
    )
    series_by_dataset[dsName] = series_by_site
    skipped_all.append(skipped_df)
    print(f"{dsName}: extracted {len(series_by_site)} sites, skipped {len(skipped_df)}")

skipped_df = pd.concat(skipped_all, ignore_index=True)

#%% build one combined dataframe per site (obs + emoElite + era5Land)

combined_site_data = {}

for site_id, df in site_data.items():
    obs_dates = pd.to_datetime(df["TIMESTAMP"]).dt.normalize()

    combined = pd.DataFrame({
        "TIMESTAMP": obs_dates.values,
        "obs": df["NETRAD"].astype(float).values,
    })
    combined = combined.set_index("TIMESTAMP")

    # attach each dataset's grid values, reindexed onto the obs dates
    # sites that were skipped for a given dataset (bad lat/lon) get an all-NaN column
    for cfg in dataset_configs:
        dsName = cfg["dsName"]
        grid_series = series_by_dataset[dsName].get(site_id)
        if grid_series is not None:
            combined[dsName] = grid_series.reindex(combined.index).values
        else:
            combined[dsName] = np.nan

    combined = combined.reset_index()  # put TIMESTAMP back as a normal column
    combined_site_data[site_id] = combined

print(f"\nBuilt combined time series for {len(combined_site_data)} sites")
if len(skipped_df) > 0:
    print("\nSites with lat/lon problems (all-NaN column for that dataset):")
    for _, row in skipped_df.iterrows():
        print(f"  {row['site_id']}: {row['reason']}")

#%% checks

for site_id, df in combined_site_data.items():
    assert "TIMESTAMP" in df.columns, f"{site_id} missing TIMESTAMP"
    assert pd.api.types.is_datetime64_any_dtype(df["TIMESTAMP"]), f"{site_id} TIMESTAMP not datetime"
    assert "obs" in df.columns, f"{site_id} missing obs column"
    for cfg in dataset_configs:
        assert cfg["dsName"] in df.columns, f"{site_id} missing {cfg['dsName']} column"
    assert len(df) == len(site_data[site_id]), f"{site_id} row count changed from original site_data"

assert set(combined_site_data.keys()) == set(site_data.keys()), "site list changed during processing"

print("All checks passed")

#%% save the dict to disk

with open('/data/saved_vars/timeseries_netRad', 'wb') as file:
    pickle.dump(combined_site_data, file)