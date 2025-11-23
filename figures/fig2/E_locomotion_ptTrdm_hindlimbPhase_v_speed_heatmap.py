import sys
from pathlib import Path
import pandas as pd
import os
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_processing, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

from mpl_toolkits.axes_grid1 import make_axes_locatable

# phase_bounds = [-0.125, 0.125, 0.375, 0.625, 0.875]
phase_bounds = [-0.1, 0.1, 0.4, 0.6, 0.9]

yyyymmdd = '2022-08-18'
df, _, _ = data_loader.load_processed_data(dataToLoad = 'strideParams', 
                                               yyyymmdd = yyyymmdd,
                                               appdx = '',
                                               outputDir = Config.paths["passiveOpto_output_folder"],
                                               limb = 'lH1')

mice = Config.passiveOpto_config['mice']

# speed_bounds = [0,25,50,75,100,125,150]
speed_bounds = utils_processing.split_by_percentile(df['speed'], 5)

hmlg_fracs = np.empty((len(phase_bounds)-1,len(speed_bounds)-1, len(mice)))*np.nan
hmlg_counts = hmlg_fracs.copy()

for im, m in enumerate(mice):
    df_m = df[df['mouseID'] == m]
    for sp in range(len(speed_bounds)-1):
        df_sub = df_m[(df_m['speed']>speed_bounds[sp]) & (df_m['speed']<speed_bounds[sp+1])]
        hmlg_phases = df_sub['rH1']
        hmlg_phases = hmlg_phases.copy()
        hmlg_phases.loc[hmlg_phases<0] +=1
        hmlg_phases.loc[hmlg_phases>0.875] -=1
        hmlg_sums = np.empty(len(phase_bounds)-1)
        
        for i in range(len(phase_bounds)-1):
           hmlg_sums[i] = ((hmlg_phases>=phase_bounds[i]) & (hmlg_phases<phase_bounds[i+1])).sum() 
         
        hmlg_counts[:, sp, im] = hmlg_sums
        hmlg_fracs[:, sp, im] = hmlg_sums/hmlg_sums.sum()
    overall_prevalences = hmlg_counts.mean(axis=2)/hmlg_counts.mean(axis=2).sum()
    
# DEFINE A COLOUR MAP
from matplotlib.colors import LinearSegmentedColormap
c1 = '#142c2c'#'#492020'
c2 = '#62b7b7'
c4 = '#e5b3bb'#'#80c7c7'
c3 = '#ddcc77'
n_bins = 100
cmap_name = 'teal_cmap'
cm = LinearSegmentedColormap.from_list(cmap_name, [c1,c2,c3,c4], N=n_bins)

# PLOT HEATMAP
fontsize = 6
fig, ax = plt.subplots(figsize = (2.8,1.6))
divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size = '5%', pad = 0.05)
cax = divider.append_axes('bottom', size = '5%', pad = 0.12)
data_to_plot = np.nanmean(hmlg_fracs,axis=2)
im = ax.imshow(data_to_plot, 
               cmap = cm, 
                vmin = 0, vmax = 1, 
               aspect = 0.6)

# marginal frequencies of limb phase bounds
counts_per_phase_bound = hmlg_counts.sum(axis=1)  # (phase bounds, mice)
freqs_per_phase_bound = counts_per_phase_bound #/ counts_per_phase_bound.sum(axis=0)  # (phase bounds, mice)
phase_freqs_mean = freqs_per_phase_bound.mean(axis=1)
phase_freqs_err = freqs_per_phase_bound.std(axis=1)/ np.sqrt(freqs_per_phase_bound.shape[1])
for i, (mean, se) in enumerate(zip(phase_freqs_mean, phase_freqs_err)):
    ax.text(data_to_plot.shape[1]-0.4,
            i,
            f"{mean:.0f}±{se:.0f}",
            va='center', ha='left',
            fontsize=fontsize-1,
            color='grey'
            )
    
# marginal frequencies of speed quintiles
counts_per_speed_quintile = hmlg_counts.sum(axis=0)  # (speed quintiles, mice)
freqs_per_speed_quintile = counts_per_speed_quintile #/ counts_per_speed_quintile.sum(axis=0)  # (speed quintiles, mice)
speed_freqs_mean = freqs_per_speed_quintile.mean(axis=1)
speed_freqs_err = freqs_per_speed_quintile.std(axis=1)/ np.sqrt(freqs_per_speed_quintile.shape[1])
for i, (mean, se) in enumerate(zip(speed_freqs_mean, speed_freqs_err)):
    ax.text(i,
            -1,
            f"{mean:.0f}\n±{se:.0f}",
            va='center', ha='center',
            fontsize=fontsize-1,
            color='grey'
            )
ax.text(data_to_plot.shape[1]+0.1, -1, 'stride\ncounts', 
        ha='center', va='center',fontsize=fontsize-1,
        color='grey')            

# prevalences
for i in range(data_to_plot.shape[0]):
    for j in range(data_to_plot.shape[1]):
        clr = 'white' if data_to_plot[i,j]<0.3 else 'black'
        ax.text(j, i, f"{data_to_plot[i,j]:.2f}",
                va='center', ha='center', color=clr,
                fontsize=fontsize-1.5)

# ticks and tick labels
ax.set_yticks(np.arange(len(phase_bounds)-1))
ax.set_yticklabels(labels = ["synchrony","R-leading","alternation","L-leading"], size =fontsize)
ax.set_xticks(np.arange(-0.5,len(speed_bounds)-1,1))
ax.set_xticklabels([f"{i:.0f}" for i in speed_bounds], size =fontsize)

# axis labels
ax.set_ylabel("Hindlimb phase", size = fontsize)
ax.set_xlabel("Speed quintiles (cm/s)", size = fontsize)

# colourbar
cbar = fig.colorbar(im, cax=cax, orientation = 'horizontal')
cbar.ax.tick_params(length=2, width=0.5, direction='out')
cbar.ax.set_xticks([0,0.2,0.4,0.6, 0.8, 1])
cbar.ax.set_xticklabels(labels = [0,0.2,0.4,0.6, 0.8, 1], size = fontsize)
cbar.ax.set_xlabel("Fraction of strides\nwithin a speed quintile", size = fontsize)
# plt.tight_layout()

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"passiveOptoTreadmill_hindlimbPhase_heatmap.svg",
            transparent = True,
            dpi =300)