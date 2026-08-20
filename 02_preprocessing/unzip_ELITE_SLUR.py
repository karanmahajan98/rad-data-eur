# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 12:46:17 2025

@author: Karan Mahajan
"""

import os 
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
INPUT_DIR = "data/ELITE_SLDR_again"     # root folder containing year subfolders 2000–2020
OUTPUT_DIR = "data/ELITE_SLDR_unzip"
NUM_WORKERS = 8                        # adjust to the number of parallel workers you want
# Tile range for Europe
H_MIN, H_MAX = 15, 22  # horizontal tile numbers
V_MIN, V_MAX = 1, 6    # vertical tile numbers
EXPECTED_COUNT = (H_MAX - H_MIN + 1) * (V_MAX - V_MIN + 1)

def process_day(zip_path):
    """
    Process a single day's ZIP: extract only the desired tiles if complete,
    otherwise return (year, julian_day) to flag as incomplete.
    """
    basename = os.path.basename(zip_path)
    year = basename[:4]
    julian_day = basename[4:7]

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # find all .h5 members matching our tile range by parsing hXXvYY
            matching = []
            for member in z.namelist():
                if not member.lower().endswith(".h5"):
                    continue
                m = re.search(r"_h(\d{2})v(\d{2})\.h5$", member)
                if not m:
                    continue
                h, v = int(m.group(1)), int(m.group(2))
                if H_MIN <= h <= H_MAX and V_MIN <= v <= V_MAX:
                    matching.append(member)

            if len(matching) >= EXPECTED_COUNT:
                # prepare output directory
                out_dir = os.path.join(OUTPUT_DIR, year, julian_day)
                os.makedirs(out_dir, exist_ok=True)
                # extract the filtered tiles
                for member in matching:
                    z.extract(member, out_dir)
                return None
            else:
                return (year, julian_day)
    except zipfile.BadZipFile:
        # treat unreadable zips as incomplete
        return (year, julian_day)

def main():
    incomplete_days = []

    # loop through years sequentially
    for year in sorted(os.listdir(INPUT_DIR)):
        print(f'Currently Processing Year : {year}')
        year_path = os.path.join(INPUT_DIR, year)
        if not os.path.isdir(year_path) or not year.isdigit():
            continue

        # gather all .zip paths for this year
        zip_paths = [
            os.path.join(year_path, fname)
            for fname in sorted(os.listdir(year_path))
            if fname.endswith(".zip")
        ]
        if not zip_paths:
            continue

        # process all days in parallel
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(process_day, zp): zp for zp in zip_paths}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    incomplete_days.append(result)

    # report incomplete days
    if incomplete_days:
        print("Incomplete data for the following year/day combinations:")
        for yr, day in sorted(incomplete_days):
            print(f"  • Year {yr}, Day {day}")
    else:
        print("All days have complete tile sets.")

if __name__ == "__main__":
    main()