#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karan mahajan

Calcualte RMSE and R2 between fluxnet observations and different radiation 
datasets (EMOELITE and ERA5-Land). 
"""
# import necessary libraries
import pandas as pd
import numpy as np
import os
import glob
import xarray as xr
import pickle

#%% load the fluxnet site data that was processed in another script (process_fluxnet.py)
with open('/data/saved_vars/site_data', 'rb') as file:
    site_data = pickle.load(file)

MIN_OBS = 90 # do not compute score if number of observations less than this val

# one entry per product we want to validate against FLUXNET
dataset_configs = [
    {"dsName": "emoElite", "folderPath": "/data/net_SW_eur_CF_F32", 
     "varName": "net_shortwave",
     "dimNames": {"time": "time", "lat": "lat", "lon": "lon"}},
    {"dsName": "era5Land", "folderPath": "/data/era5_land_rad", 
     "varName": "ssr",
     "dimNames": {"time": "time", "lat": "lat", "lon": "lon"}},
]

#%%

def open_ds(dsName, folderPath, varName, dimNames):
    
    """
    Opens a netCDF dataset using xarray
    
    Parameters
    ----------
    dsName : str
        user defined name of the dataset file.
    folderPath: str
        path of the folder where netCDF files are stored
    varName: str 
        name of the array variable in the netCDF files
    dimNames: dict
        A dictionary to map the dimension names in the netCDF files to standard dim names
    

    Returns
    -------
    ds: xarray dataset
    grid_time: pd.datetime
       the time coordinates in the stitched dataset
    lon_max: maximum longitude value: to check if 360 degree system
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

#%%
def compute_scores(obs, grid):
    
    """
    Calculate rmse and r-squared between observations and the datasets
    """
    rmse = np.sqrt(np.mean((grid - obs) ** 2))

    ss_res = np.sum((obs - grid) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return r2, rmse

#%%

def score_dataset(dsName, ds, grid_time, lon_max, varName, dimNames, site_data, min_obs=MIN_OBS):
    
    """
    loop over sites for a single dataset, 
    extract nearest cell, 
    calcualte performance score
    
    Parameters
    ----------
    dsName : str
        user defined name of the dataset file.
    ds: xarray dataset
        loaded dataset from open_ds()
    grid_time: pd.datetime
       the time coordinates in the stitched dataset
    lon_max: float
        maximum longitude value: to check if 360 degree system   
    varName: str 
        name of the array variable in the netCDF files
    dimNames: dict
        A dictionary to map the dimension names in the netCDF files to standard dim names
    site_data: dict
        dict where keys give site_id and values are timeseries for different rad vars as a dataframe
    min_obs: int
        Minimum number of obervations to calculate the score
    

    Returns
    -------
    results: dataframe
        Dataframe with site score information (id, score, n_obs used for scoring) 
    skipped: dataframe
       Sites that were skipped and not processed   
    """
    
    
    results = []
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

        obs = pd.Series(
            df["net_SW"].astype(float).values,
            index=pd.to_datetime(df["TIMESTAMP"]).dt.normalize(),
            name="obs",
        )

        # find the nearest cell in the dataset to the site
        cell = ds[varName].sel(**{dimNames["lat"]: lat, dimNames["lon"]: lon_q}, method="nearest")
        grid = pd.Series(cell.values.astype(float), index=grid_time, name="grid")

        paired = pd.concat([obs, grid], axis=1, join="inner").dropna()

        if len(paired) < min_obs:
            skipped.append({"site_id": site_id,
                            "reason": f"[{dsName}] only {len(paired)} overlapping valid obs (<{min_obs})"})
            continue

        r2, rmse = compute_scores(paired["obs"].values, paired["grid"].values)

        if not (np.isfinite(r2) and np.isfinite(rmse)):
            skipped.append({"site_id": site_id, "reason": f"[{dsName}] score came out NaN/inf"})
            continue

        results.append({
            "site_id": site_id,
            f"r2_{dsName}": r2,
            f"rmse_{dsName}": rmse,
            f"n_obs_{dsName}": len(paired),
        })

    return pd.DataFrame(results), pd.DataFrame(skipped)

#%% open both datasets

opened_datasets = {}
for cfg in dataset_configs:
    ds, grid_time, lon_max = open_ds(cfg["dsName"], cfg["folderPath"], cfg["varName"], cfg["dimNames"])
    opened_datasets[cfg["dsName"]] = {"ds": ds, "grid_time": grid_time, "lon_max": lon_max}
    
#%%  run score_dataset() once per product, then merge into one table

scores_by_dataset = {}
skipped_all = []

for cfg in dataset_configs:
    dsName = cfg["dsName"]
    print(f'..........Processing dataset: {dsName}.........\n')
    opened = opened_datasets[dsName]
    scores_df, skipped_df = score_dataset(
        dsName, opened["ds"], opened["grid_time"], opened["lon_max"],
        cfg["varName"], cfg["dimNames"], site_data,
    )
    scores_by_dataset[dsName] = scores_df
    skipped_all.append(skipped_df)
    print(f"{dsName}: scored {len(scores_df)} sites, skipped {len(skipped_df)}")

# outer merge so a site kept for one product but skipped for the other still shows up,
# just with NaN in that product's columns
merged_scores = scores_by_dataset[dataset_configs[0]["dsName"]]
for cfg in dataset_configs[1:]:
    merged_scores = merged_scores.merge(scores_by_dataset[cfg["dsName"]], on="site_id", how="outer")

skipped_df = pd.concat(skipped_all, ignore_index=True)

# attach site metadata (same for every product, so build it once)
metadata_rows = []

for site_id, df in site_data.items():
    metadata_rows.append({
        "site_id": site_id,
        "SITE_NAME": df["SITE_NAME"].iloc[0],
        "LOCATION_ELEV": df["LOCATION_ELEV"].iloc[0],
        "LOCATION_LAT": df["LOCATION_LAT"].iloc[0],
        "LOCATION_LONG": df["LOCATION_LONG"].iloc[0],
        "IGBP": df["IGBP"].iloc[0],
    })
metadata_df = pd.DataFrame(metadata_rows)

sorted_df = merged_scores.merge(metadata_df, on="site_id", how="left")

print(f"\nFinal table has {len(sorted_df)} sites")
if len(skipped_df) > 0:
    print("\nSites/products where scoring was not possible:")
    for _, row in skipped_df.iterrows():
        print(f"  {row['site_id']}: {row['reason']}")
        
#%% Some checks to see if all went well

for cfg in dataset_configs:
    dsName = cfg["dsName"]
    r2_col, rmse_col, n_obs_col = f"r2_{dsName}", f"rmse_{dsName}", f"n_obs_{dsName}"

    assert sorted_df[r2_col].dropna().between(0, 1).all(), f"{r2_col} has values outside [0, 1]"
    assert (sorted_df[rmse_col].dropna() >= 0).all(), f"{rmse_col} has negative values"
    assert (sorted_df[n_obs_col].dropna() >= MIN_OBS).all(), f"{n_obs_col} has a site below the min-obs cutoff"

    # every site should either have a score for this dataset, or a skip reason for it
    n_scored = sorted_df[r2_col].notna().sum()
    n_skipped_this_ds = skipped_df["reason"].str.contains(f"[{dsName}]", regex=False).sum()
    assert n_scored + n_skipped_this_ds == len(site_data), \
        f"{dsName}: scored + skipped counts don't match total sites"

print("All checks passed")

#%% Adding country column and sort by site_id

country_lookup = {
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland", "CZ": "Czech Republic",
    "DE": "Germany", "DK": "Denmark", "ES": "Spain", "FI": "Finland", "FR": "France",
    "IE": "Ireland", "IT": "Italy", "NL": "Netherlands", "NO": "Norway", "SE": "Sweden",
}

sorted_df["country"] = sorted_df["site_id"].str[:2].map(country_lookup)

missing_country = sorted_df[sorted_df["country"].isna()]
if len(missing_country) > 0:
    print("These site ids didn't match a country, add them to country_lookup:")
    print(missing_country["site_id"].tolist())

sorted_df = sorted_df.sort_values("site_id").reset_index(drop=True)

sorted_df['dataset'] = 'netSW'

#%%

with open('/data/saved_vars/scores_netSW', 'wb') as file:
    pickle.dump(sorted_df, file)
