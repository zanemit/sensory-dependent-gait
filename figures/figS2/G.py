import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader, utils_math, utils_processing
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandlerDouble, AnyObjectHandler

param = 'speed'
strideparam = 'strideFreq'
group_num = 5
limbs = {'rH0': 'homologous', 'lF0': 'homolateral', 'rF0': 'diagonal'}
limbRef = 'lH1'
configs = {"passiveOpto": Config.passiveOpto_config, 
           "mtTreadmill": Config.mtTreadmill_config}

# SPLIT DATA IN QUINTILES
df, _, _ = data_loader.load_processed_data(outputDir = Config.paths[f"{list(configs.keys())[0]}_output_folder"], 
                                            dataToLoad = "strideParams", 
                                            yyyymmdd = '2022-08-18',
                                            appdx = "",
                                            limb = "COMBINED")

param_split = utils_processing.split_by_percentile(df['speed'], 5)

# PLOT
fig, ax = plt.subplots(1, 1, figsize=(1.3, 1.3))

legend_colours = np.empty((4, 0)).tolist()
legend_linestyles = np.empty((4, 0)).tolist()

total_mice = np.sum([len(configs[x][mouse_str]) for x, mouse_str in zip (list(configs.keys())*2, ['mice', 'mice_level', 'mice', 'mice_incline'])])
stride_params = np.empty((total_mice, group_num, 4, 2)) # mice, speed groups, setups, alternating/synchronous
stride_params[:] = np.nan
                                              
for ix, (yyyymmdd, appdx, mice_str, clr_id, lnst, cfg_id) in enumerate(zip(
        ['2022-08-18', '2022-08-18', '2021-10-23', '2022-05-06'],
        ["", "_incline", "", ""],
        ["mice", "mice", "mice_level", "mice_incline"],
        [1, 3, 1, 3],
        ['solid', 'solid', 'dashed', 'dashed'],
        [0,0,1,1])):                                   
    df, _, _ = data_loader.load_processed_data(outputDir = Config.paths[f"{list(configs.keys())[cfg_id]}_output_folder"], 
                                                dataToLoad = "strideParams", 
                                                yyyymmdd = yyyymmdd,
                                                appdx = appdx,
                                                limb = limbRef)
    
    synchronous = np.where((df['rH0'] >= -0.1) & (df['rH0'] <= 0.1))[0] #synchrony
    alternating = np.where((df['rH0'] >= 0.4) | (df['rH0'] <= -0.4))[0] #alternation
    mice = configs[list(configs.keys())[cfg_id]][mice_str]
    
    for im, m in enumerate(mice):
        df_sub = df[df['mouseID'] == m]
        df_grouped = df_sub.groupby(pd.cut(df_sub[param], param_split)) 
        group_row_ids = df_grouped.groups
        for c, gaitrows in enumerate([alternating, synchronous]):
            grouped_dict = {key:df_sub.loc[np.intersect1d(val, gaitrows),strideparam].values for key,val in group_row_ids.items()} 
            keys = list(grouped_dict.keys())
            for i, gkey in enumerate(keys):
                values = grouped_dict[gkey][~np.isnan(grouped_dict[gkey])]
                if values.shape[0] < (Config.passiveOpto_config["stride_num_threshold"]):
                    print(f"Mouse {m} data excluded from group {gkey} because there are only {values.shape[0]} valid trials!")
                    continue
                
                stride_params[im, i, ix, c] = np.nanmean(grouped_dict[gkey])
    
    
    for ig, (arr, clr) in enumerate(zip([stride_params[:,:,:,0], stride_params[:,:,:,1]],
                                           ['homologous', 'greys'])):      #alternation = light teal/grey, synchrony = dark teal/grey  
        sem = scipy.stats.sem(np.nanmean(arr, axis = 2), axis = 0, nan_policy = 'omit') 
        ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., arr.shape[0]-1)   
        ax.fill_between(np.arange(group_num), 
                        np.nanmean(np.nanmean(arr, axis = 2), axis = 0) - ci, 
                        np.nanmean(np.nanmean(arr, axis = 2), axis = 0) + ci, 
                        facecolor = FigConfig.colour_config[clr][clr_id], alpha = 0.2)
        ax.plot(np.arange(group_num),  
                np.nanmean(np.nanmean(arr, axis = 2), axis = 0), 
                color = FigConfig.colour_config[clr][clr_id], 
                linestyle = lnst, 
                linewidth = 1)
        legend_colours[clr_id-1 + ig].append(FigConfig.colour_config[clr][clr_id])
        legend_linestyles[clr_id-1 + ig].append(lnst) 

for ix, (clr, clr_id, ig, ploc) in enumerate(zip(
        ['homologous', 'homologous', 'greys', 'greys'],
        [1, 3, 1, 3],
        [0, 1, 0, 1],
        [3.5,2.8,2,1.2])):      
    #alt no incline, sync no incline, alt incline, sync incline
    t, p = scipy.stats.ttest_ind(stride_params[:,:,0,ig],
                                 stride_params[:,:,1,ig],
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
            ax.text(ip, ploc-0.2, p_text, ha = 'center', color = FigConfig.colour_config[clr][clr_id])

ax.set_ylim(1,13)        
ax.set_ylabel("Stride frequency (strides/s)")
ax.set_yticks([1,5,9,13])
ax.set_xlim(-0.5,group_num-1)
ax.set_xlabel('Speed (cm/s)')
xticklabels = [f'({k.left:.0f},{k.right:.0f}]' for k in group_row_ids.keys()]
ax.set_xticks(np.arange(group_num))
ax.set_xticklabels(xticklabels, rotation = 20, ha = 'right')

lgd_actors = [(legend_colours[i], legend_linestyles[i]) for i in range(ix+1)]
        
lgd = fig.legend(lgd_actors,
                ['alternating, no incline', 'synchronous, no incline',
                 'alternating, incline', 'synchronous, incline'],
                handler_map={tuple: AnyObjectHandlerDouble()}, loc = 'upper left',
                bbox_to_anchor=(0,-0.52,1,0.3), mode="expand", borderaxespad=0.1,
                ncol = 1) 

lgd_actors2 = [(list(np.unique(legend_colours)), np.unique(legend_linestyles)[i]) for i in range(len(np.unique(legend_linestyles)))]

lgd2 = fig.legend([(lgd_actors2[0][0], "dashed"), (lgd_actors2[1][0], "solid")],
                ['motorised treadmill', 'passive treadmill'],
                handler_map={tuple: AnyObjectHandler()}, loc = 'upper left',
                bbox_to_anchor=(0.12,0.75,0.7,0.2), mode="expand", borderaxespad=0.1,
                ncol = 1, frameon = False) 

figtitle = f"{yyyymmdd}_strideFreq_comparison.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight')