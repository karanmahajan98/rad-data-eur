#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: karan mahajan

Extract fluxnet zip files using multiple workers.
Process fluxnet radiation dataset. 
calculate performance of geospatial datasets at fluxnet sites.
"""
#%% Import necessary libraries
import os
import re
import zipfile
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle
#%% extract paths to the fluxnet zip files

zip_dir = "/data/FLUXNET/raw_zips"              
extracted_dir = "/data/FLUXNET/extracted_zips"  
os.makedirs(extracted_dir, exist_ok=True)

YEAR_START, YEAR_END = 2001, 2020
MIN_OBS = 90

RAD_COLS = ["TIMESTAMP", "SW_IN_F", "LW_IN_F", "SW_OUT", "LW_OUT", "NETRAD"]
BIF_VARS = ["SITE_NAME", "LOCATION_ELEV", "LOCATION_LAT", "LOCATION_LONG", "IGBP"]

zip_paths = [os.path.join(zip_dir, f) for f in os.listdir(zip_dir) if f.endswith(".zip")]
print(f"Found {len(zip_paths)} zip files")

#%% guess site ID using file name and re expression

def guess_site_id(filename):
    """
    Extract site ID from fluxnet daily file using re.

    Parameters
    ----------
    filename : str
        name of the daily data file.

    Returns
    -------
    site_ID or filename: int
    """
    match = re.search(r"([A-Z]{2}-[A-Za-z0-9]{2,6})", os.path.basename(filename))
    return match.group(1) if match else os.path.basename(filename)

#%%

def process_one_zip(zip_path):
    """
    Worker function that handles a single zip file. For each zip ,pull out the 
    daily FLUXMET file and save it to disk,
    and read the BIF metadata file straight from the zip.
    
    Parameters
    ----------
    zip_path : str
        Path of zip file that will be extracted.

    Returns
    -------
    result: dict
        Dictionary giving various information on the respective fluxnet file   
    """
    
    
    result = {"zip": zip_path, "site_id": guess_site_id(zip_path),
              "dd_path": None, "meta": {v: np.nan for v in BIF_VARS}, "status": "ok"}

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        dd_name = next((n for n in names if "_FLUXMET_DD_" in n), None)
        bif_name = next((n for n in names if "_FLUXNET_BIF_" in n), None)

        if dd_name is None:
            result["status"] = "no_daily_file"
            return result

        # save the daily file to extracted folder
        out_path = os.path.join(extracted_dir, os.path.basename(dd_name))
        with zf.open(dd_name) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        result["dd_path"] = out_path

        # site id from the daily file name is more reliable than from the zip name
        site_match = re.search(r"([A-Z]{2}-[A-Za-z0-9]{2,6})_FLUXNET", dd_name)
        if site_match:
            result["site_id"] = site_match.group(1)

        # grab metadata if the BIF file exists, otherwise we just keep the NaNs
        if bif_name is not None:
            with zf.open(bif_name) as f:
                bif_df = pd.read_csv(f)
            for var in BIF_VARS:
                match_row = bif_df.loc[bif_df["VARIABLE"] == var, "DATAVALUE"]
                if len(match_row) > 0:
                    result["meta"][var] = match_row.iloc[0]

    return result

#%%

def process_daily_file(record):
    """
    process each extracted daily file
    sites with some missing radiation columns are still kept
    TIMESTAMP is super important column so leave sites without this

    Parameters
    ----------
    result: dict
        Dictionary giving various information on the respective fluxnet file 

    Returns
    -------
    df: pandas dataframe
        Dataframe with the extracted information
        
    n_obs: int
        number of valid observations extracted from the fluxnet site/file
    
    """
    # check the header first, without loading the whole file
    header_cols = pd.read_csv(record["dd_path"], nrows=0).columns.tolist()

    if "TIMESTAMP" not in header_cols:
        return None, None, "missing TIMESTAMP column"

    cols_available = [c for c in RAD_COLS if c in header_cols]
    cols_missing = [c for c in RAD_COLS if c not in header_cols]

    df = pd.read_csv(record["dd_path"], usecols=cols_available)
    df = df.replace(-9999, np.nan)

    # add back any missing radiation columns as all-NaN so the dataframe
    # always has the same shape downstream
    for col in cols_missing:
        df[col] = np.nan

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%Y%m%d")
    df = df[(df["TIMESTAMP"].dt.year >= YEAR_START) & (df["TIMESTAMP"].dt.year <= YEAR_END)]
    df = df.reset_index(drop=True)

    # net_SW / net_LW = incoming minus outgoing (positive = downward, following FLUXNET convention)
    # only calcualte where both in and out values are available
    both_sw = df["SW_IN_F"].notna() & df["SW_OUT"].notna()
    df["net_SW"] = np.where(both_sw, df["SW_IN_F"] - df["SW_OUT"], np.nan)

    both_lw = df["LW_IN_F"].notna() & df["LW_OUT"].notna()
    df["net_LW"] = np.where(both_lw, df["LW_IN_F"] - df["LW_OUT"], np.nan)

    n_obs = {
        "net_SW": int(df["net_SW"].notna().sum()),
        "net_LW": int(df["net_LW"].notna().sum()),
        "NETRAD": int(df["NETRAD"].notna().sum()),
    }
    return df, n_obs, None

#%% implement the extraction logic using parallel workers

with ThreadPoolExecutor(max_workers=8) as pool:
    extraction_results = list(pool.map(process_one_zip, zip_paths))

# just a check to see which sites did not have a daily file
no_daily_file = [r for r in extraction_results if r["status"] == "no_daily_file"]
ok_results = [r for r in extraction_results if r["status"] == "ok"]

print(f"{len(ok_results)} sites had a daily file, {len(no_daily_file)} did not")

#%% Loop over the sites, attach metadata, exclude sites with less than 90 observations

site_data = {}
excluded_rows = []  # for exluded sites

for record in ok_results:
    site_id = record["site_id"]
    df, n_obs, error_reason = process_daily_file(record)

    if error_reason is not None:
        excluded_rows.append({"site_id": site_id, "reason": error_reason})
        continue

    keep_site = max(n_obs.values()) >= MIN_OBS
    if not keep_site:
        reason = f"fails 90-obs rule (net_SW={n_obs['net_SW']}, net_LW={n_obs['net_LW']}, NETRAD={n_obs['NETRAD']})"
        excluded_rows.append({"site_id": site_id, "reason": reason})
        continue

    # attach metadata, same value repeated for every row
    for var in BIF_VARS:
        df[var] = record["meta"][var]

    site_data[site_id] = df

# add the sites that never had a daily file in the first place
for record in no_daily_file:
    excluded_rows.append({"site_id": record["site_id"], "reason": "no daily FLUXMET file in zip"})

excluded_sites_df = pd.DataFrame(excluded_rows)

print(f"Kept {len(site_data)} sites, excluded {len(excluded_sites_df)} sites")

#%% just some basic checks so we notice if something got dropped by accident

assert len(site_data) + len(excluded_sites_df) == len(zip_paths), "site counts should add up"

for site_id, df in site_data.items():
    for col in RAD_COLS + ["net_SW", "net_LW"] + BIF_VARS:
        assert col in df.columns, f"{site_id} is missing column {col}"
    assert df["TIMESTAMP"].is_unique, f"{site_id} has duplicate TIMESTAMP values"
    n_obs = {
        "net_SW": df["net_SW"].notna().sum(),
        "net_LW": df["net_LW"].notna().sum(),
        "NETRAD": df["NETRAD"].notna().sum(),
    }
    assert max(n_obs.values()) >= MIN_OBS, f"{site_id} shouldn't have passed the 90-obs rule"

print("All checks passed")

#%% save site data, to be used later

with open('/data/saved_vars/site_data', 'wb') as file:
    pickle.dump(site_data, file)
