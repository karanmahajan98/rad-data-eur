# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:52:45 2025

@author: Karan Mahajan

Mosaic all the MODIS tiles for a single day in a single raster, reproject, clip and save as TIF file
Do this for all the days
"""

import os
import glob
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio import MemoryFile
from affine import Affine
from osgeo import gdal
import ast
from concurrent.futures import ThreadPoolExecutor
from time import time



dataset_name = 'SLUR_daily'   # dataset name in HDF5 files
fill_value    = 999           # fill-value indicating missing data
number_workers = 4           # parallel number
min_lon = -25.241666666666667
min_lat = 22.7583333333333328 
max_lon = 50.241666666666660
max_lat = 72.241666666666674 
output_bounds=(min_lon, min_lat, max_lon, max_lat) # for clipping with gdal
num_tiles = 48 # number of tiles for the study area. If less than this,  flag the day with incomplete tiles
not_open_files = [] # global list for files that couldn't be opened

# Define functions

def get_geotransform_from_h5(filename):
    """
    Gets the geotransform from a single .h5 file

    Parameters
    ----------
    filename : string
        DESCRIPTION.

    Returns
    -------
    geotrans_list : list

    """
    try:
        
        with h5py.File(filename, 'r') as f:
            geotrans_str = f.attrs['geotrans']
            geotrans_list  = ast.literal_eval(geotrans_str)
        return geotrans_list
    except OSError:
        not_open_files.append(filename)
        return 'not_open'


def read_tile_array_from_h5(filename, dataset_name, fill_value=None):
    """
    Reads a .h5 file as an array. Implements the necessary scaling factor.

    Parameters
    ----------
    filename : string
    
    dataset_name : string
        dataset name in HDF5 files.
    fill_value : int, optional
        NaN fill value defined in the dataset. The default is None.

    Returns
    -------
    arr : numpy array

    """
    #print(filename)
    try: 
        with h5py.File(filename, 'r') as f:
            
            arr = f[dataset_name][()]
            if fill_value is not None:
                arr = arr.astype(np.float32)
                arr[arr == fill_value] = np.nan
                arr = (arr *0.05) -1000 # Scale value according to ELITE documentation
        return arr
    except OSError:
        not_open_files.append(filename)
        return 'not_open'
    

def mosaic_tiles_h5(filenames, dataset_name, fill_value=None):
    """
    Mosaics all the .h5 tile files for a single day and puts them as an in memory raster
    This in memory raster will be reprojected and clipped in another function

    Parameters
    ----------
    filenames : list
        list of file names (paths) for all the .h5 tiles for a single day.
    dataset_name : string
        as defined above.
    fill_value : int, optional
        NaN fill value defined in the dataset. The default is None.

    Returns
    -------
    memfile : .tif
        in memory tif file.

    """
    # 1. Extract all arrays and geotransforms
    tiles = []
    geotransforms = []
    shapes = []
    for fname in filenames:
        arr = read_tile_array_from_h5(fname, dataset_name, fill_value=fill_value)
        gt = get_geotransform_from_h5(fname)
        
        if (type(arr) == str) or (type(gt) == str):
            return 'not_open'
        
        tiles.append(arr)
        geotransforms.append(gt)
        shapes.append(arr.shape)
    
    # 2. Find mosaic bounds
    bounds = []
    for arr, gt in zip(tiles, geotransforms):
        #print(gt)
        #print(type(gt))
        ulx, xres, _, uly, _, yres = gt
        height, width = arr.shape
        lrx = ulx + width * xres
        lry = uly + height * yres
        bounds.append((ulx, lrx, uly, lry))
    min_ulx = min(b[0] for b in bounds)
    max_lrx = max(b[1] for b in bounds)
    max_uly = max(b[2] for b in bounds)
    min_lry = min(b[3] for b in bounds)

    # 3. Assume all have same pixel size
    xres = geotransforms[0][1]
    yres = geotransforms[0][5]
    dtype = tiles[0].dtype
    fill = np.nan if fill_value is not None else 0

    # 4. Compute mosaic shape
    mosaic_width = int(np.ceil((max_lrx - min_ulx) / xres))
    mosaic_height = int(np.ceil((max_uly - min_lry) / abs(yres)))
    mosaic = np.full((mosaic_height, mosaic_width), fill, dtype=np.float32)
    
    # 5. Place each tile
    for arr, gt in zip(tiles, geotransforms):
        ulx, _, _, uly, _, _ = gt
        row_off = int(round((max_uly - uly) / abs(yres)))
        col_off = int(round((ulx - min_ulx) / xres))
        h, w = arr.shape
        mosaic[row_off:row_off+h, col_off:col_off+w] = arr
    
    # 6. Prepare the transform for the mosaic
    mosaic_transform = Affine.from_gdal(min_ulx, xres, 0, max_uly, 0, yres)
    
    # 7. Write to GeoTIFF
    memfile = MemoryFile()
    with memfile.open(
        driver='GTiff',
        height=mosaic.shape[0],
        width=mosaic.shape[1],
        count=1,
        dtype='float32',
        crs='+proj=sinu +a=6371007.181 +b=6371007.181 +units=m',
        transform=mosaic_transform,
        nodata=np.nan
    ) as dataset:
        dataset.write(mosaic, 1)
    #print("Mosaic created successfully in memory.")

    return memfile

def memfile_to_tif(output_raster, h5_files, dataset_name, output_bounds, fill_value):
    """
    Reprojects and clips in memory raster and saves it as tif file

    Parameters
    ----------
    output_raster : string
        location for output raster.
    h5_files : list
        files paths for all .h5 tiles for a single day.
    dataset_name : string
        as defined above.
    fill_value : int
        as defined above.
    output_bounds : tuple
        Clipping extent

    Returns
    -------
    None. Just saves .tif file to disk

    """
    in_memory_mosaic = mosaic_tiles_h5(h5_files, dataset_name, fill_value)
    if type(in_memory_mosaic) == str:
        return 'not_open'
    src_dataset = in_memory_mosaic.open()
    in_memory_path = src_dataset.name
    warp = gdal.Warp(output_raster, in_memory_path, dstSRS='EPSG:4326',  outputBounds = output_bounds)
    #print(f"Reprojected mosaic saved to disk at: {output_raster}")
    warp = None
    src_dataset.close()
    in_memory_mosaic.close()
    return 'opened'


def process_day(args):
    """
    Worker function to process a single day
    Uses module-level dataset_name and fill_value
    """
    year_str, day_dir, out_dir = args
    day = os.path.basename(day_dir)
    h5_files = glob.glob(os.path.join(day_dir, '*.h5'))
    # Mark incomplete if not 30 tiles
    if len(h5_files) != num_tiles:
        print(f"Incomplete files for year and day: {year}, {day}")
        return (year_str, day)
    

    out_name = f"SLUR_{year_str}{day}.tif"
    out_path = os.path.join(out_dir, out_name)
    # Skip if already exists
    if os.path.exists(out_path):
        return None

    status = memfile_to_tif(out_path, h5_files, dataset_name, output_bounds, fill_value)
    if status == 'not_open':
        print(f"ERROR: Cannot open file for year: {year_str} and day: {day}")
        return (year_str, day)   #This later appends to incomplete day list

    # Log progress every 50 days
    if int(day) % 50 == 0:
        print(f"Year {year_str} – Day {day} processed.")
    return None

if __name__ == "__main__":
    t0 = time()
    # Configure input/output
    input_base = "data/ELITE_SLUR"
    output_base = "data/ELITE_SLUR_tifs"
    incomplete = []
    for year in range(2001, 2021):
        year_str = str(year)
        print(f"Processing year {year_str}")
        in_year = os.path.join(input_base, year_str)
        out_year = os.path.join(output_base, year_str)
        os.makedirs(out_year, exist_ok=True)

        day_dirs = sorted(
            d for d in glob.glob(os.path.join(in_year, '*'))
            if os.path.isdir(d) and os.path.basename(d).isdigit()
        )
        tasks = [(year_str, d, out_year) for d in day_dirs]

        # Parallel execution with 4 processes
        with ThreadPoolExecutor(max_workers=number_workers) as executor:
            for result in executor.map(process_day, tasks):
                if result:
                    incomplete.append(result)
        #print(f"Completed year {year_str}")
    print(f"Time taken to save all tif files = {(time()-t0)/3600} hours")    
    print("Incomplete folders:", incomplete)
    print("Files that could not be opened:", not_open_files)