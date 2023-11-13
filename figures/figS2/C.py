import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\SensoryDependentGait")

from processing import data_loader, utils_processing
from processing.data_config import Config
from figures_benzon.fig_config import Config as FigConfig
from figures_benzon.fig_config import AnyObjectHandlerDouble, AnyObjectHandler

param = 'speed'
group_num = 5
limb = 'rH0'
limbRef = 'lH1'
yyyymmdd =  '2021-10-23'
appdx = ''
mice_str = "mice_level"
clr = 'greys'
clr_id = 0
ploc = 0.7
cfg_id = 0

# SPLIT DATA IN QUINTILES
df, _, _ = data_loader.load_processed_data(outputDir = Config.paths[f"mtTreadmill_output_folder"], 
                                            dataToLoad = "strideParams", 
                                            yyyymmdd = yyyymmdd,
                                            appdx = "",
                                            limb = "COMBINED")

param_split = utils_processing.split_by_percentile(df['speed'], 5)
# param_split = [9,19,29,39,49,59]

# ARR DEFINITION
mouse_len =  len(Config.mtTreadmill_config[mice_str])
arr = np.empty((mouse_len, group_num, 2)) # mouse_num, group_num, sync/alt, mtTRDM/passiveOPTO
arr[:] = 0

# PLOT
fig, ax = plt.subplots(1, 1, figsize=(3.25, 3.25))

legend_colours = np.empty((2, 0)).tolist()
legend_linestyles = np.empty((2, 0)).tolist()

for ix, (yyyymmdd, appdx, mice_str) in enumerate(zip(
        ['2021-10-23', '2022-05-06'],
        ["", ""],
        ["mice_level", "mice_incline"]
        )):                                        
    df, _, _ = data_loader.load_processed_data(outputDir = Config.paths["mtTreadmill_output_folder"], 
                                                dataToLoad = "strideParams", 
                                                yyyymmdd = yyyymmdd,
                                                appdx = appdx,
                                                limb = limbRef)
    
    # split param into groups (across mice, across conditions)
    # param_noOutliers = utils_processing.remove_outliers(df[param])
    # param_split = np.linspace(param_noOutliers.min(), param_noOutliers.max(), group_num+1)
    
    # find where limb/limbRef are alternating and synchronous
    alternating = np.where((df[limb] >= 0.4) | (df[limb] <= -0.4))[0]
    synchronous = np.where((df[limb] >= -0.1) & (df[limb] <= 0.1))[0]    
    
    for im, m in enumerate(Config.mtTreadmill_config[mice_str]):
        df_sub = df[df['mouseID'] == m]
        
        # xvals = [np.nanmean((a,b,)) for a,b in zip(param_split[:-1], param_split[1:])]
        df_grouped = df_sub.groupby(pd.cut(df_sub[param], param_split)) 
        group_row_ids = df_grouped.groups
        # print([len(group_row_ids[key]) for key in group_row_ids.keys()])
        arr[im, :, 0] += [len(np.intersect1d(alternating, group_row_ids[key]))/len(group_row_ids[key]) if len(group_row_ids[key])>0 else np.nan for key in group_row_ids.keys()]
        arr[im, :, 1] += [len(np.intersect1d(synchronous, group_row_ids[key]))/len(group_row_ids[key]) if len(group_row_ids[key])>0 else np.nan for key in group_row_ids.keys()]

arr = arr/2
print(group_row_ids.keys())
for i, (lnst, tlt) in enumerate(zip(['solid', 'dashed'],
                                    ['alternating', 'synchronous'])):
    meanval = np.nanmean(arr[:,:,i], axis = 0)
    print(f'{tlt}: {meanval}')
    sem = scipy.stats.sem(arr[:,:,i], axis = 0, nan_policy = 'omit')  
    ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., arr[:,:,i].shape[0]-1) 
    ax.fill_between(np.arange(group_num), 
                    meanval - ci,
                    meanval + ci, 
                    facecolor = FigConfig.colour_config[clr][clr_id], 
                    alpha = 0.2)
    
    ax.plot(np.arange(group_num),
            np.nanmean(arr[:,:,i], axis = 0),
            color = FigConfig.colour_config[clr][clr_id],  
            linewidth = 2,
            linestyle = lnst,
            label = tlt
            )
    
    legend_colours[i] = FigConfig.colour_config[clr][clr_id]
    legend_linestyles[i] = lnst 
    
t, p = scipy.stats.ttest_ind(arr[:,:,0],
                             arr[:,:,1],
                             axis = 0, 
                             equal_var =False, 
                             nan_policy = 'omit')
pvals = np.empty((group_num), dtype = 'object')
for ip, pval in enumerate(p.data):
    p_text = ('*' * (pval < np.asarray(FigConfig.p_thresholds)).sum())
    if (pval < np.asarray(FigConfig.p_thresholds)).sum() == 0 and not np.isnan(pval):
        p_text = "n.s."
        ax.text(ip, ploc+0.005, p_text, ha = 'center', color = FigConfig.colour_config[clr][clr_id])
    else:
        ax.text(ip, ploc, p_text, ha = 'center', color = FigConfig.colour_config[clr][clr_id])

ax.set_ylim(-0.05,0.8)

ax.set_xlabel('Speed quintile')
xticklabels = [1,2,3,4,5]
ax.set_xticks(np.arange(group_num))
ax.set_xlim(-0.5,group_num-1)
ax.set_xticklabels(xticklabels)

ax.set_ylabel("Fraction of strides")
lgd_actors2 = [(list(np.unique(legend_colours)), np.unique(legend_linestyles)[i]) for i in range(len(np.unique(legend_linestyles)))]

lgd2 = fig.legend([(lgd_actors2[0][0], "dashed"), (lgd_actors2[1][0], "solid")],
                ['synchrony', 'alternation'],
                handler_map={tuple: AnyObjectHandler()}, loc = 'upper left',
                bbox_to_anchor=(0.3,0.9,0.5,0.2), mode="expand", borderaxespad=0.1,
                ncol = 1)#, frameon = False) 


plt.tight_layout()

figtitle = f"BENZON_{yyyymmdd}_strict_gait_fractions_homologous_mtTreadmill.svg"
    
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
                dpi = 300, 
                bbox_extra_artists = (lgd2, ), 
                bbox_inches = 'tight'
                )


