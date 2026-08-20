# rad-data-eur
Python code used to create and validate 20-years net shortwave and net radiation datasets for Europe at 1-arcminue spatial and daily temporal resolution.


## Repository structure

```text
net-radiation-europe-pipeline/
├── README.md
├── LICENSE
├── environment.yaml
│
├── 01_data_acquisition/
│   ├── download_fluxnet.ipynb
│
├── 02_preprocessing/
│   ├── process_fluxnet.py
│   ├── unzip_ELITE_SLUR.py 
│   ├── ELITE_h5_to_tiff.py
|   ├── PAR_clip_regrid.py
|   ├── GLASS_alb_clip_ffill.py
|   ├── era5_land_hourly_to_daily.py
|   ├── extract_timeseries_netLW.py
|   └── extract_timeseries_netRad.py
│
├── 03_radiation_calculation/
│   ├── calc_blue_sky_alb.py
│   ├── calc_netSW.py
│   ├── calc_netLW.py
│   └── calc_netRad.py
│
├── 04_cf_compliance_export/
│   ├── netSW_to_CF.py
│   └── netRad_to_CF.py
│
└── 05_validation_analysis/
    ├── calc_score.py
    ├── calc_score_lw.py
    ├── plot_scores.py
    ├── error_decompose_resid.py
    ├── fig_error_decompose_resid.py
    ├── RMSE_netRad_ERA5_land.py
    └── RMSE_netSW_ERA5_land.py
