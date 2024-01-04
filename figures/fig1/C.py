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

yyyymmdd = '2021-10-26'
param = 'headHW'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                        dataToLoad='forceplateData_CoMgrouped', 
                                        yyyymmdd = yyyymmdd, 
                                        appdx = param)

new_line = '\n'
plt.rcParams['axes.titlepad'] = 10
mice = np.unique(df.columns.get_level_values(1))
conditions = np.unique(df.columns.get_level_values(0))[[0,-1]]
cond_titles = [f"{h}{new_line}{c[:5]},{c.split(',')[-1][:5]}]" for h,c in zip(['head low', 'head high'],conditions)]

# MEANS WITH SHADED 95% CI
fig, ax = plt.subplots(1, conditions.shape[0], 
                       figsize=(conditions.shape[0]*1.45, 1.3), #1.5
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
                      color=FigConfig.colour_config["greys7"][mnum], zorder=2)
        ax[icon].plot(np.nanmean(df_sub.loc[:, (slice(None),'CoMx')], axis=1), 
                      np.nanmean(df_sub.loc[:, (slice(None),'CoMy')], axis=1), 
                      color=FigConfig.colour_config["main"], 
                      linewidth=2, 
                      zorder=3)

    ax[icon].set_xlabel('Mediolateral\ncentre of support')
    ax[icon].xaxis.labelpad = 10
    ax[icon].set_title(cond_t)
ax[0].set_ylim(-1, 1)
ax[0].set_yticks([])
ax[0].set_ylabel('Anteroposterior\ncentre of support')
ax[0].yaxis.labelpad = 15
fig.text(0,0.85,"Weight-\nadjusted\nhead\nheight", ha = 'center')

grey_selection = [FigConfig.colour_config["greys7"][i] for i in [1,3,5]]
lgd = fig.legend([(grey_selection,"solid"), 
                ([FigConfig.colour_config["main"]],"solid")], 
                ["trials\nby\nmouse", "per-\nmouse\naverage"], handler_map={tuple: AnyObjectHandler()}, 
                loc = 'upper center', bbox_to_anchor=(0.95,0.3,0.23,0.5),#(0.1,-0.35,0.8,0.2),
                mode="expand", borderaxespad=0, ncol = 1)

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'MS2_{yyyymmdd}_CoMxy_over_time_{param}_allMice.svg'), 
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight',
            transparent = True)
