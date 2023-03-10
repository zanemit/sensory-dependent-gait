import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig

yyyymmdd = '2022-08-18'
outputDir = Config.paths['passiveOpto_output_folder']
limb = 'lF0'
appdx = ''
param = 'snoutBodyAngle'
lbl = 'Snout-hump angle (deg)'


fig, ax = plt.subplots(2,3,figsize = (1.7*3,1.5), gridspec_kw = {'height_ratios': [9,1], 'width_ratios': [15,15,15]}, sharex = 'col')
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])

for i, limb in enumerate(['lF0','rF0','rH0']):
    cot_pvals = pd.read_csv(os.path.join(outputDir, yyyymmdd + f'_COTpvals_{limb}_{param}{appdx}.csv'), index_col =0)
    cot = pd.read_csv(os.path.join(outputDir, yyyymmdd + f'_COT_{limb}_{param}{appdx}.csv'), index_col =0)
    cot_ptext = pd.read_csv(os.path.join(outputDir, yyyymmdd + f'_COTptext_{limb}_{param}{appdx}.csv'), index_col =0)
    cot_uni_pvals = pd.read_csv(os.path.join(outputDir, yyyymmdd + f'_COT_UNIFORMpvals_{limb}_{param}{appdx}.csv'), index_col =0)
    cot_uni = pd.read_csv(os.path.join(outputDir, yyyymmdd + f'_COT_UNIFORM_{limb}_{param}{appdx}.csv'), index_col =0)
    cot_uni_ptext = pd.read_csv(os.path.join(outputDir, yyyymmdd + f'_COT_UNIFORMptext_{limb}_{param}{appdx}.csv'), index_col =0)
     
    cot_ptext = np.asarray(["" if type(x) == float else x for x in np.asarray(cot_ptext).flatten()]).reshape(cot.shape[0], cot.shape[1])
    cot_uni_ptext = np.asarray(["" if type(x) == float else x for x in np.asarray(cot_uni_ptext).flatten()]).reshape(cot_uni.shape[0], cot_uni.shape[1])
    
    title = f'head height trials:\nbody tilt, {limb[:2]}'
    
    if i < 2:          
        sns.heatmap(cot, 
                    vmin=0, 
                    vmax=0.2, 
                    ax = ax[0, i], 
                    cbar = False, 
                    cmap = 'mako',
                    annot = cot_ptext, 
                    fmt = "", 
                    annot_kws = {"size":6}
                    )
      
    else: 
        sns.heatmap(cot, 
                    vmin=0, 
                    vmax=0.2, 
                    cbar_ax = cbar_ax,
                    ax = ax[0, i], 
                    cmap = 'mako',
                    annot = cot_ptext, 
                    fmt = "", 
                    annot_kws = {"size":6}
                    )    
    
    ax[0,i].set_title(title)
    
    sns.heatmap(cot_uni.T, 
                vmin=0, 
                vmax=0.2, 
                ax = ax[1,i],
                cmap = 'mako',
                cbar = False, 
                annot = cot_uni_ptext.T, 
                fmt = "", 
                annot_kws = {"size":6})
    
    for tick in ax[1,i].get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha('right')
    
    if i != 0:             
        ax[0,i].tick_params(labelleft = False)
        ax[1,i].tick_params(labelleft = False)
    
ax[1,0].set_yticklabels(['uniform'], rotation = 0)
fig.text(0.5, -0.24, lbl, ha='center') # xlabel
ax[0,0].set_ylabel(lbl)
fig.subplots_adjust(wspace = 0.3)
cbar_ax.set_ylabel('Circular optimal transport')
cbar_ax.set_yticks([0,0.1,0.2])
# plt.tight_layout()

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_cotAcrossMicePermuted_HH_ALL.svg", dpi = 300)
   