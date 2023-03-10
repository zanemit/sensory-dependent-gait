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

outputDir = Config.paths['mtTreadmill_output_folder']
limb = 'lF0'


fig, ax = plt.subplots(2,2,figsize = (1.3*2,1.5), gridspec_kw = {'height_ratios': [9,1], 'width_ratios': [1,1]}, sharex = 'col')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.84, 0.18, 0.01, 0.7])
stattype = 'WASSERSTEIN'
param = 'snoutBodyAngle'
lbl = 'Snout-hump angle (deg)'
ticklabels = ['[0, 0.4π]','(0.4π, 0.8π]','(0.8π, 1.2π]','(1.2π, 1.6π]']

for i, (yyyymmdd, title) in enumerate(zip(['2021-10-23','2022-05-06'],
                                          ['level locomotion:\nbody tilt', 'incline locomotion:\nbody tilt']
                                          )):
    cot_pvals = pd.read_csv(os.path.join(outputDir, f'{yyyymmdd}_{stattype}pvals_{param}_{limb}.csv'), index_col =0)
    cot = pd.read_csv(os.path.join(outputDir, f'{yyyymmdd}_{stattype}_{param}_{limb}.csv'), index_col =0)
    cot_ptext = pd.read_csv(os.path.join(outputDir, f'{yyyymmdd}_{stattype}ptext_{param}_{limb}.csv'), index_col =0)
    cot_uni_pvals = pd.read_csv(os.path.join(outputDir, f'{yyyymmdd}_{stattype}_UNIFORMpvals_{param}_{limb}.csv'), index_col =0)
    cot_uni = pd.read_csv(os.path.join(outputDir, f'{yyyymmdd}_{stattype}_UNIFORM_{param}_{limb}.csv'), index_col =0)
    cot_uni_ptext = pd.read_csv(os.path.join(outputDir, f'{yyyymmdd}_{stattype}_UNIFORMptext_{param}_{limb}.csv'), index_col =0)
     
    cot_ptext = np.asarray(["" if type(x) == float else x for x in np.asarray(cot_ptext).flatten()]).reshape(cot.shape[0], cot.shape[1])
    cot_uni_ptext = np.asarray(["" if type(x) == float else x for x in np.asarray(cot_uni_ptext).flatten()]).reshape(cot_uni.shape[0], cot_uni.shape[1])
     
    if i < 1:          
        sns.heatmap(cot, 
                    vmin=0, 
                    vmax=8, 
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
                    vmax=8, 
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
                vmax=8, 
                ax = ax[1,i],
                cmap = 'mako',
                cbar = False, 
                annot = cot_uni_ptext.T, 
                fmt = "", 
                annot_kws = {"size":6})
    
    # for tick in ax[1,i].get_xticklabels():
    #     tick.set_rotation(35)
    #     tick.set_ha('right')
    ax[1,i].set_xticklabels(ticklabels, rotation = 35, ha = 'right')
    # ax[1,i].set_xlabel(lbl)
    ax[i,1].tick_params(labelleft = False)#, left = False)
                 
ax[0,0].set_ylabel(lbl)
fig.text(0.5, -0.24, lbl, ha='center') # xlabel
ax[1,0].set_yticklabels(['uniform'], rotation = 0)
ax[0,0].set_yticklabels(ticklabels, rotation = 0)
fig.subplots_adjust(wspace = 0.4)
cbar_ax.set_ylabel('Wasserstein distance')
cbar_ax.set_yticks([0,2,4,6,8])
# plt.tight_layout()

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_wassersteinAcrossMicePermuted_{limb}.svg", dpi = 300)
   