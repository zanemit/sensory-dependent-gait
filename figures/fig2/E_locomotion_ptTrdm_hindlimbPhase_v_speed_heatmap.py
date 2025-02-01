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

phase_bounds = [-0.125, 0.125, 0.375, 0.625, 0.875]

pt_path = r"C:\Users\MurrayLab\Documents\PassiveOptoTreadmill\2022-08-18_strideParams_lH1.csv"
df = pd.read_csv(pt_path)

mice = Config.passiveOpto_config['mice']

# speed_bounds = [0,25,50,75,100,125,150]
speed_bounds = utils_processing.split_by_percentile(df['speed'], 5)

hmlg_fracs = np.empty((len(phase_bounds)-1,len(speed_bounds)-1, len(mice)))*np.nan

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
          
        hmlg_fracs[:, sp, im] = hmlg_sums/hmlg_sums.sum()
    
# DEFINE A COLOUR MAP
from matplotlib.colors import LinearSegmentedColormap
c1 = '#142c2c'#'#492020'
c2 = '#e5b3bb'#'#80c7c7'
n_bins = 100
cmap_name = 'teal_cmap'
cm = LinearSegmentedColormap.from_list(cmap_name, [c1,c2], N=n_bins)

# PLOT
fontsize = 6
fig, ax = plt.subplots(figsize = (2.3,1.3))
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size = '5%', pad = 0.05)
im = ax.imshow(np.nanmean(hmlg_fracs,axis=2), cmap = cm, vmin = 0, vmax = 0.8, aspect = 0.6)
ax.set_yticks(np.arange(len(phase_bounds)-1))
# ax.set_yticklabels(labels = ["[-0.125π, 0.125π)","[0.125π, 0.375π)","[0.375π, 0.625π)","[0.625π, 0.875π)"], size =fontsize)
ax.set_yticklabels(labels = ["synchrony","R-leading","alternation","L-leading"], size =fontsize)
ax.set_xticks(np.arange(-0.5,len(speed_bounds)-1,1))
# ax.set_xticklabels([f"[{speed_bounds[i]}-{speed_bounds[i+1]})" for i in range(len(speed_bounds)-1)], rotation = 30, size =fontsize)
ax.set_xticklabels([f"{i:.0f}" for i in speed_bounds], size =fontsize)
ax.set_ylabel("Hindlimb phase", size = fontsize)
ax.set_xlabel("Speed quintiles (cm/s)", size = fontsize)
cbar = fig.colorbar(im, cax=cax, orientation = 'vertical')
cbar.ax.set_yticks([0,0.2,0.4,0.6, 0.8])
cbar.ax.set_yticklabels(labels = [0,0.2,0.4,0.6,0.8], size = fontsize)
cbar.ax.set_ylabel("Fraction of strides", size = fontsize)
plt.tight_layout()

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"passiveOptoTreadmill_hindlimbPhase_heatmap.svg",
            transparent = True,
            dpi =300)