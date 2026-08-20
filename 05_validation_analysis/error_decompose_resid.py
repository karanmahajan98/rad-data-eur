#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karan mahajan

Decompose the MSE of net radiation into shortwave, longwave and cross terms,
to check which component drives the error in the merged Rn product.

Identity used (errors are additive because Rn = SWnet + LWnet):
    MSE(Rn) = MSE(SW) + MSE(LW) + 2*mean(e_SW * e_LW)
"""
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

#%% settings and load the saved dicts

PATHS = {
    'sw': '/home/mahajan/paper_net_rad/saved_vars/timeseries_netSW',
    'lw': '/home/mahajan/paper_net_rad/saved_vars/timeseries_netLW',
    'rn': '/home/mahajan/paper_net_rad/saved_vars/timeseries_netRad',
}

# column name of our product in each dict (it is not the same in all three)
PROD_COL = {'sw': 'emoElite', 'lw': 'Elite', 'rn': 'emoElite'}

# physically plausible daily means, used to throw out corrupted records
LIMITS = {'sw': (-50, 400), 'lw': (-250, 100), 'rn': (-250, 500)}

MIN_OBS = 90
OUT_CSV = '/home/mahajan/paper_net_rad/saved_vars/mse_decomposition_per_site_resid.csv'

#%% load data

ts = {}
for key, path in PATHS.items():
    with open(path, 'rb') as f:
        ts[key] = pickle.load(f)
    print(f'{key}: {len(ts[key])} sites')

common_sites = sorted(set(ts['sw']) & set(ts['lw']) & set(ts['rn']))
print(f'sites present in all three dicts: {len(common_sites)}')

#%% build one aligned table per site

def build_site_table(site):
    """
    Merge the three radiation components onto a common date index and keep only days
    where every series is present and physically plausible.
    
    Parameters
    ----------
    site : str
        FLUXNET site ID.

    Returns
    -------
    m: dataframe
        combined dataframe for given site with all rad components.
    """
    a = ts['sw'][site].set_index('TIMESTAMP')
    b = ts['lw'][site].set_index('TIMESTAMP')
    c = ts['rn'][site].set_index('TIMESTAMP')

    m = pd.DataFrame({
        'sw_obs': a['obs'], 'sw_prod': a[PROD_COL['sw']], 'sw_era': a['era5Land'],
        'lw_obs': b['obs'], 'lw_prod': b[PROD_COL['lw']], 'lw_era': b['era5Land'],
        'rn_obs': c['obs'], 'rn_prod': c[PROD_COL['rn']], 'rn_era': c['era5Land'],
    })
    m = m.dropna()

    for comp, (lo, hi) in LIMITS.items():
        for suffix in ['obs', 'prod', 'era']:
            col = f'{comp}_{suffix}'
            m = m[m[col].between(lo, hi)]

    return m

#%%

def decompose(e_sw, e_lw, resid):
    """MSE of the sum, split into the three terms. Returns a dict.
    
    Parameters
    ----------
    e_sw : numpy.ndarray
        Daily shortwave error for a single site.
    e_lw : numpy.ndarray
        Daily longwave error for a single site. 
    resid : numpy.ndarray
        difference between the actual NETRAD and the sum of SW and LW obs

    Returns
    -------
    : dict
        Dict of final error decomposed into individual errors.
    """
    mse_sw = np.mean(e_sw ** 2)
    mse_lw = np.mean(e_lw ** 2)
    cross = 2 * np.mean(e_sw * e_lw)
    resid_term = np.mean(resid ** 2) - 2 * np.mean(resid * (e_sw + e_lw))
    return {'mse_sw': mse_sw, 'mse_lw': mse_lw, 'cross': cross, 'resid_term': resid_term,
            'mse_rn': mse_sw + mse_lw + cross + resid_term}

#%% main loop

rows = []
skipped = []

for site in common_sites:         # loop through each site
    m = build_site_table(site)    # combine all three products in a single df

    if len(m) < MIN_OBS:          # need minimum 90 days to do the analysis
        skipped.append({'site': site, 'reason': f'only {len(m)} usable days'})
        continue

    # residual: difference between measured net radiation and the sum of sw and lw components
    resid = (m['rn_obs'] - (m['sw_obs'] + m['lw_obs'])).values

    e_sw = (m['sw_prod'] - m['sw_obs']).values   # calculating the errors: used later for also calculating MSE
    e_lw = (m['lw_prod'] - m['lw_obs']).values
    E_sw = (m['sw_era'] - m['sw_obs']).values     # capital E for ERA5-Land, small e for our product
    E_lw = (m['lw_era'] - m['lw_obs']).values

    ours = decompose(e_sw, e_lw, resid)
    era5 = decompose(E_sw, E_lw, resid)
    
    true_ours = np.mean((m['rn_prod'].values - m['rn_obs'].values) ** 2) # true RMSE calculated directly
    true_era5 = np.mean((m['rn_era'].values - m['rn_obs'].values) ** 2) # should be same as the RMSE calculated using indiv components

    # the identity must hold exactly, otherwise something is wrong
    assert np.isclose(ours['mse_rn'], true_ours), site
    assert np.isclose(era5['mse_rn'], true_era5), site

    rows.append({
        'site': site, 'country': site[:2], 'n': len(m),
        'resid_rmse': np.sqrt(np.mean(resid ** 2)),   # here we are comparing net rad obs with sw_net + lw_net
        'resid_bias': resid.mean(),
        'rn_obs_sd': m['rn_obs'].std(),
        # fractions of MSE(Rn), our product
        'f_sw': ours['mse_sw'] / ours['mse_rn'],
        'f_lw': ours['mse_lw'] / ours['mse_rn'],
        'f_cross': ours['cross'] / ours['mse_rn'],
        # same for era5
        'f_sw_era': era5['mse_sw'] / era5['mse_rn'],
        'f_lw_era': era5['mse_lw'] / era5['mse_rn'],
        'f_cross_era': era5['cross'] / era5['mse_rn'],
        'f_resid': ours['resid_term'] / ours['mse_rn'],
        'f_resid_era': era5['resid_term'] / era5['mse_rn'],
        # attribution of the gap between the two products
        'd_sw': ours['mse_sw'] - era5['mse_sw'],
        'd_lw': ours['mse_lw'] - era5['mse_lw'],
        'd_cross': ours['cross'] - era5['cross'],
        'd_resid': ours['resid_term'] - era5['resid_term'],
        'gap': ours['mse_rn'] - era5['mse_rn'],
        # rmse for reporting
        'rmse_sw': np.sqrt(ours['mse_sw']), 'rmse_sw_era': np.sqrt(era5['mse_sw']),
        'rmse_lw': np.sqrt(ours['mse_lw']), 'rmse_lw_era': np.sqrt(era5['mse_lw']),
        'rmse_rn': np.sqrt(ours['mse_rn']), 'rmse_rn_era': np.sqrt(era5['mse_rn']),
        # error correlation between components
        'corr_err': np.corrcoef(e_sw, e_lw)[0, 1],
        'corr_err_era': np.corrcoef(E_sw, E_lw)[0, 1],
        # normalised rmse
        'nrmse_sw': np.sqrt(ours['mse_sw']) / m['sw_obs'].std(),
        'nrmse_lw': np.sqrt(ours['mse_lw']) / m['lw_obs'].std(),
        'nrmse_sw_era': np.sqrt(era5['mse_sw']) / m['sw_obs'].std(),
        'nrmse_lw_era': np.sqrt(era5['mse_lw']) / m['lw_obs'].std(),
    })

d = pd.DataFrame(rows)
d.to_csv(OUT_CSV, index=False)
print(f'\n{len(d)} sites usable, {len(skipped)} skipped')

#%% check 1: is the energy balance residual small enough to trust the decomposition?

print('\n===== CLOSURE RESIDUAL: NETRAD_obs - (SWnet_obs + LWnet_obs) =====')
print(f"median RMSE  : {d.resid_rmse.median():.2f} W/m2")
print(f"as % of Rn SD: {100 * (d.resid_rmse / d.rn_obs_sd).median():.1f} %")
print(f"median bias  : {d.resid_bias.median():+.2f} W/m2")
print(f"sites < 10 W/m2: {(d.resid_rmse < 10).sum()} / {len(d)}")

if (d.resid_rmse / d.rn_obs_sd).median() > 0.2:
    print('WARNING: residual is large, decomposition is against summed components only')

#%% check 2: the decomposition itself

print('\n===== MSE DECOMPOSITION (fractions of MSE Rn, median over sites) =====')
print(f"our product : SW {d.f_sw.mean():.2f}   LW {d.f_lw.mean():.2f}   cross {d.f_cross.mean():+.2f}   resid {d.f_resid.mean():+.2f}")
print(f"ERA5-Land   : SW {d.f_sw_era.mean():.2f}   LW {d.f_lw_era.mean():.2f}   cross {d.f_cross_era.mean():+.2f}   resid {d.f_resid_era.mean():+.2f}")

#%% check 3: attribute the gap between the two products
# means are used here because MSE terms add up, medians do not

print('\n===== ATTRIBUTION OF MSE GAP (ours - ERA5), W2/m4 =====')
print(f"shortwave term : {d.d_sw.mean():+8.0f}")
print(f"longwave term  : {d.d_lw.mean():+8.0f}")
print(f"cross term     : {d.d_cross.mean():+8.0f}")
print(f"residual term  : {d.d_resid.mean():+8.0f}")
print(f"total gap      : {d.gap.mean():+8.0f}")
print(f"(sum check     : {d.d_sw.mean() + d.d_lw.mean() + d.d_cross.mean():+8.0f})")
print(f"sites where our Rn is worse: {(d.gap > 0).sum()} / {len(d)}")

#%% check 4: rmse comparison with significance test

print('\n===== RMSE (W/m2), median over sites =====')
for comp, label in [('sw', 'net SW'), ('lw', 'net LW'), ('rn', 'Rn')]:
    diff = d[f'rmse_{comp}'] - d[f'rmse_{comp}_era']
    stat, p = wilcoxon(diff)
    print(f"{label:7s} ours {d[f'rmse_{comp}'].median():6.2f}  ERA5 {d[f'rmse_{comp}_era'].median():6.2f}  "
          f"diff {diff.median():+6.2f}  p={p:.1e}  ours better at {(diff < 0).sum()}/{len(d)}")

print('\n===== NORMALISED RMSE (RMSE / SD of observed component) =====')
print(f"net SW: ours {d.nrmse_sw.median():.2f}  ERA5 {d.nrmse_sw_era.median():.2f}")
print(f"net LW: ours {d.nrmse_lw.median():.2f}  ERA5 {d.nrmse_lw_era.median():.2f}")

print('\n===== ERROR CORRELATION BETWEEN SW AND LW =====')
print(f"ours {d.corr_err.median():+.2f}   ERA5-Land {d.corr_err_era.median():+.2f}")

#%% per country summary

print('\n===== MEDIAN RMSE BY COUNTRY =====')
g = d.groupby('country').agg(
    n=('site', 'count'),
    sw=('rmse_sw', 'median'), sw_era=('rmse_sw_era', 'median'),
    lw=('rmse_lw', 'median'), lw_era=('rmse_lw_era', 'median'),
    rn=('rmse_rn', 'median'), rn_era=('rmse_rn_era', 'median'))
print(g.round(1).to_string())