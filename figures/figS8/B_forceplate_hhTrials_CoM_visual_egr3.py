import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt


sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

yyyymmdd = '2023-11-06'
param = 'headHW'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                        dataToLoad='forceplateData_CoMgrouped', 
                                        yyyymmdd = yyyymmdd, 
                                        appdx = param)

grey_clrs = ["#4d4d4d", "#595959", "#666666", "#707070", "#7a7a7a", "#858585",
             "#919191", "#9e9e9e", "#a8a8a8", "#b2b2b2", "#bfbfbf", "#cccccc", 
             "#d9d9d9", "#e5e5e5"] 

new_line = '\n'
plt.rcParams['axes.titlepad'] = 10
mice = np.unique(df.columns.get_level_values(1))
conditions = np.unique(df.columns.get_level_values(0))[[0,-1]]
cond_titles = ['head low', 'head high']

# MEANS WITH SHADED 95% CI
fig, ax = plt.subplots(1, conditions.shape[0], 
                       figsize=(conditions.shape[0]*1, 0.8), #1.5
                       sharey=True)

for icon, (cond, cond_t) in enumerate(zip(conditions, cond_titles)):  # loop over levels/intervals
    ax[icon].spines['left'].set_color('none')
    ax[icon].spines['bottom'].set_color('none')
    ax[icon].set_xlim(-1, 1)
    # ax[icon].set_yticklabels([-1, 1])
    ax[icon].set_xticks([])
    # ax[icon].set_xticklabels([-1, 1])
    ax[icon].plot([-1, 1], [1, -1], color='grey',linestyle=':', zorder=1)
    ax[icon].plot([-1, 1], [-1, 1], color='grey',linestyle=':', zorder=1)
    ax[icon].plot([-1, 1], [-1, -1], color='grey',linestyle=':', zorder=1) # bottom horizontal
    ax[icon].plot([-1, -1], [-1, 1], color='grey',linestyle=':', zorder=1) # left vertical
    ax[icon].plot([1, -1], [1, 1], color='grey',linestyle=':', zorder=1) # top horizontal
    ax[icon].plot([1, 1], [1, -1], color='grey',linestyle=':', zorder=1) # right vertical
    
    ax[icon].text(-1,-1, "(-1,-1)", ha = 'center', va = 'top')
    ax[icon].text(1,-1, "(1,-1)", ha = 'center', va = 'top')
    ax[icon].text(1,1, "(1,1)", ha = 'center', va = 'bottom')
    ax[icon].text(-1,1, "(-1,1)", ha = 'center', va = 'bottom')
    
    for mnum, mouse in enumerate(mice):
        try:
            df_sub = df.loc[:, (cond, mouse)]
        except:
            continue
        ax[icon].plot(df_sub.loc[:, (slice(None),'CoMx')], 
                      df_sub.loc[:, (slice(None),'CoMy')],
                      color=grey_clrs[mnum], zorder=2)
        ax[icon].plot(np.nanmean(df_sub.loc[:, (slice(None),'CoMx')], axis=1), 
                      np.nanmean(df_sub.loc[:, (slice(None),'CoMy')], axis=1), 
                      color=FigConfig.colour_config["main"], 
                      linewidth=1.5, 
                      zorder=3)

    ax[icon].set_xlabel('Mediolateral\ncentre of support')
    ax[icon].xaxis.labelpad = 10
    ax[icon].set_title(cond_t)
ax[0].set_ylim(-1, 1)
ax[0].set_yticks([])
ax[0].set_ylabel('Anteroposterior\ncentre of support')
ax[0].yaxis.labelpad = 15

grey_selection = [grey_clrs[i] for i in [1,3,5]]
lgd = fig.legend([(grey_selection,"solid"), 
                (grey_clrs,"solid")], 
                ["trials by mouse", "per-mouse average"], handler_map={tuple: AnyObjectHandler()}, 
                loc = 'upper center', bbox_to_anchor=(-0.05,-0.75,1,0.5),#(0.1,-0.35,0.8,0.2),
                mode="expand", borderaxespad=0, ncol = 2)

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_CoMxy_over_time_headHW_dtx.svg'), 
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight',
            transparent = True)
