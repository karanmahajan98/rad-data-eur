#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karan mahajan

Figure: Error decomposition analysis
Four panels:
 (a) paired per-site RMSE, ours vs ERA5, for the three components
 (b) MSE budget of Rn, split into SW / LW / cross terms, for both products
 (c) attribution of the MSE gap (ours - ERA5) into the same three terms
 (d) error correlation between SW and LW errors, both products
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#from scipy.stats import wilcoxon

CSV = '/home/mahajan/paper_net_rad/saved_vars/mse_decomposition_per_site_resid.csv'
OUT = '/home/mahajan/paper_net_rad/figures/fig_error_propagation_final'

d = pd.read_csv(CSV)

# ---- style ----
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 7.5,
    'axes.labelsize': 8,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 400,
})

C_OURS = '#1182D1'   # warm  = this study
C_ERA  = '#D92929'   # cool  = ERA5-Land
C_SW   = '#E0A458'
C_LW   = '#7C93C3'
C_CR   = '#9AA0A6'
GREY   = '#4A4A4A'

fig = plt.figure(figsize=(7.2, 5.9))
gs = fig.add_gridspec(2, 2, hspace=0.72, wspace=0.30,
                      left=0.085, right=0.985, top=0.93, bottom=0.085)

# ============================================================
# (a) paired RMSE per site, three components
# ============================================================
ax = fig.add_subplot(gs[0, 0])

comps = [('sw', 'Net SW'), ('lw', 'Net LW'), ('rn', r'$R_n$')]
xpos = [0, 1, 2]
off = 0.17

for i, (c, lab) in enumerate(comps):
    o = d[f'rmse_{c}'].values
    e = d[f'rmse_{c}_era'].values
    # one thin line per site connecting the two products
    for oi, ei in zip(o, e):
        col = C_OURS if oi < ei else C_ERA
        ax.plot([i - off, i + off], [oi, ei], color=col, lw=0.35, alpha=0.30,
                solid_capstyle='round', zorder=1)
    # medians
    ax.plot([i - off, i + off], [np.median(o), np.median(e)],
            color='k', lw=1.6, zorder=4, solid_capstyle='round')
    ax.scatter([i - off], [np.median(o)], s=26, color=C_OURS, ec='k', lw=0.7, zorder=5)
    ax.scatter([i + off], [np.median(e)], s=26, color=C_ERA, ec='k', lw=0.7, zorder=5)

    # significance annotation
    diff = d[f'rmse_{c}'] - d[f'rmse_{c}_era']
#    _, p = wilcoxon(diff)
    nbet = int((diff < 0).sum())
    ax.text(i, 76.5, f'{nbet}/{len(d)} sites', ha='center', va='bottom',
            fontsize=6.8, color=GREY)
#    ptxt = 'p < 0.001' if p < 1e-3 else f'p = {p:.3f}'
#    ax.text(i, 71.5, ptxt, ha='center', va='bottom', fontsize=6.2, color=GREY,
#            style='italic')

ax.set_xticks(xpos)
ax.set_xticklabels([l for _, l in comps])
ax.set_xlim(-0.55, 2.55)
ax.set_ylim(8, 86)
ax.set_ylabel(r'RMSE vs FLUXNET (W m$^{-2}$)')
ax.set_title('a)', loc='left', fontweight='bold')
ax.text(0.5, 1.045, 'Per-site error, this study vs ERA5-Land',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=8)

handles = [Line2D([], [], color=C_OURS, marker='o', mec='k', mew=0.7, ls='none',
                  ms=5, label='This study'),
           Line2D([], [], color=C_ERA, marker='o', mec='k', mew=0.7, ls='none',
                  ms=5, label='ERA5-Land')]
ax.legend(handles=handles, loc='lower left', frameon=False, ncol=2, handletextpad=0.3,
          borderpad=0.1, labelspacing=0.2, bbox_to_anchor=(0.2, -0.38))

# ============================================================
# (b) MSE budget of Rn for each product
# ============================================================
ax = fig.add_subplot(gs[0, 1])

# use means so the three terms add to the total exactly
budget = {
    'This study': [d.f_sw.mean(), d.f_lw.mean(), d.f_cross.mean()],
    'ERA5-Land': [d.f_sw_era.mean(), d.f_lw_era.mean(), d.f_cross_era.mean()],
}

labels = ['SW term', 'LW term', 'cross term']
colors = [C_SW, C_LW, C_CR]
ypos = [1, 0]

for yi, (name, vals) in zip(ypos, budget.items()):
    # SW and LW stacked on the upper bar
    left_pos = 0.0
    for v, col in zip(vals[:2], colors[:2]):
        ax.barh(yi + 0.13, v, left=left_pos, height=0.30, color=col,
                ec='white', lw=0.8, zorder=3)
        ax.text(left_pos + v / 2, yi + 0.13, f'{v:.2f}', ha='center', va='center',
                fontsize=6.8, color='k', zorder=4)
        left_pos += v
    # cross term shown separately, extending back from the SW+LW total to 1.0
    v = vals[2]
    ax.barh(yi - 0.19, v, left=left_pos, height=0.15, color=C_CR,
            ec='white', lw=0.6, zorder=3)
    ax.annotate('', xy=(left_pos + v, yi - 0.19), xytext=(left_pos, yi - 0.19),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=0.7,
                                shrinkA=0, shrinkB=0), zorder=5)
    ax.text(left_pos + v / 2, yi - 0.40, f'{v:.2f}', ha='center', va='center',
            fontsize=6.8, color=GREY, zorder=4)

ax.axvline(1.0, color='k', ls=(0, (3, 2)), lw=0.9, zorder=5)
ax.text(1.0, 1.55, r'MSE($R_n$) = 1', ha='center', va='bottom', fontsize=6.8, color=GREY)

ax.set_yticks([y + 0.05 for y in ypos])
ax.set_yticklabels(['This study', 'ERA5-Land'])
ax.set_xlim(0, 1.72)
ax.set_ylim(-0.85, 1.75)
ax.set_xlabel(r'Fraction of MSE($R_n$)')
ax.set_title('b)', loc='left', fontweight='bold')
ax.text(0.5, 1.045, r'Where the $R_n$ error comes from',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=8)

hb = [mpl.patches.Patch(fc=C_SW, ec='white', label='SW'),
      mpl.patches.Patch(fc=C_LW, ec='white', label='LW'),
      mpl.patches.Patch(fc=C_CR, ec='white', label='cross (cancelling)')]
ax.legend(handles=hb, loc='upper center', frameon=False, ncol=3,
          handletextpad=0.35, borderpad=0.1, columnspacing=0.9,
          bbox_to_anchor=(0.5, -0.28))

# ============================================================
# (c) attribution of the MSE gap
# ============================================================
ax = fig.add_subplot(gs[1, 0])

terms = [('d_sw', 'SW', C_SW), ('d_lw', 'LW', C_LW), ('d_cross', 'cross', C_CR)]
vals = [d[t].mean() for t, _, _ in terms]
gap = d.gap.mean()

# waterfall
run = 0.0
for i, ((t, lab, col), v) in enumerate(zip(terms, vals)):
    ax.bar(i, v, bottom=run, width=0.62, color=col, ec='k', lw=0.6, zorder=3)
    # connector
    if i < len(terms):
        ax.plot([i - 0.31, i + 0.31], [run + v, run + v], color=GREY, lw=0.6,
                zorder=2)
        if i < len(terms) - 1:
            ax.plot([i + 0.31, i + 1 - 0.31], [run + v, run + v], color=GREY,
                    lw=0.6, ls=(0, (2, 2)), zorder=2)
    va = 'bottom' if v > 0 else 'top'
    ax.text(i, run + v + (16 if v > 0 else -16), f'{v:+.0f}', ha='center', va=va,
            fontsize=7, fontweight='bold')
    run += v

ax.bar(3, gap, width=0.62, color='none', ec='k', lw=1.1, hatch='...', zorder=3)
ax.text(3, gap + 14, f'{gap:+.0f}', ha='center', va='bottom', fontsize=7,
        fontweight='bold')
ax.plot([2 + 0.31, 3 - 0.31], [run, run], color=GREY, lw=0.6, ls=(0, (2, 2)),
        zorder=2)

ax.axhline(0, color='k', lw=0.8, zorder=4)
ax.set_xticks(range(4))
ax.set_xticklabels(['SW\nterm', 'LW\nterm', 'cross\nterm', 'net\ngap'])
ax.set_ylabel(r'$\Delta$MSE, this study $-$ ERA5-Land' '\n' r'(W$^2$ m$^{-4}$)')
ax.set_ylim(-395, 400)
ax.set_title('c)', loc='left', fontweight='bold')
ax.text(0.5, 1.045, r'What drives the $R_n$ gap',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=8)

ax.text(0.97, 0.035, 'below 0 = we improve on ERA5-Land', transform=ax.transAxes,
        fontsize=6.4, color=GREY, style='italic', ha='right')

# ============================================================
# (d) error correlation between components
# ============================================================
ax = fig.add_subplot(gs[1, 1])

parts = ax.violinplot([d.corr_err.values, d.corr_err_era.values],
                      positions=[0, 1], widths=0.62, showextrema=False,
                      showmedians=False)
for pc, col in zip(parts['bodies'], [C_OURS, C_ERA]):
    pc.set_facecolor(col)
    pc.set_alpha(0.30)
    pc.set_edgecolor(col)
    pc.set_linewidth(0.8)

rng = np.random.default_rng(0)
for i, (v, col) in enumerate([(d.corr_err.values, C_OURS),
                              (d.corr_err_era.values, C_ERA)]):
    jit = rng.uniform(-0.085, 0.085, len(v))
    ax.scatter(i + jit, v, s=7, color=col, alpha=0.55, lw=0, zorder=3)
    med = np.median(v)
    ax.plot([i - 0.23, i + 0.23], [med, med], color='k', lw=1.6, zorder=5,
            solid_capstyle='round')
    ax.text(i + 0.28, med, f'{med:.2f}', va='center', fontsize=7,
            fontweight='bold')

ax.axhline(0, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(['This study', 'ERA5-Land'])
ax.set_xlim(-0.5, 1.62)
ax.set_ylabel(r'corr($e_{SW}$, $e_{LW}$)')
ax.set_ylim(-0.85, 0.42)
ax.set_title('d)', loc='left', fontweight='bold')
ax.text(0.5, 1.045, 'Do the two errors cancel?',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=8)

ax.text(0.03, 0.05, 'more negative = stronger cancellation', transform=ax.transAxes,
        fontsize=6.4, color=GREY, style='italic')

fig.savefig(f'{OUT}.png', bbox_inches='tight', facecolor='white')

print('saved', OUT)