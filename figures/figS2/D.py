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
    
for ix, (yyyymmdd, appdx, mice_str, clr, clr_id, cfg_id, ploc) in enumerate(zip(
        ['2022-08-18', '2022-08-18', '2021-10-23', '2022-05-06'],
        ["", "_incline", "", ""],
        ["mice", "mice", "mice_level", "mice_incline"],
        ['homolateral', 'homolateral', 'greys', 'greys'],
        [1,3,1,4],
        [0,0,1,1],
        [-0.01,-0.04,-0.07,-0.1])):                                   
    df, _, _ = data_loader.load_processed_data(outputDir = Config.paths[f"{list(configs.keys())[cfg_id]}_output_folder"], 
                                                dataToLoad = "strideParams", 
                                                yyyymmdd = yyyymmdd,
                                                appdx = appdx,
                                                limb = limbRef)
    import pickle
    idealGaitDict = pickle.load(open(Path(Config.paths["passiveOpto_output_folder"]) / "idealisedGaitDict.pkl", "rb" ))
    
    mice = configs[list(configs.keys())[cfg_id]][mice_str]
    
    trot_cots = np.empty((len(mice), group_num, len(limbs)))
    bound_cots = np.empty((len(mice), group_num, len(limbs)))
    trot_cots[:] = np.nan
    bound_cots[:] = np.nan
    
    for il, (limb, limbtitle) in enumerate(zip(limbs.keys(), limbs.values())):
        trot = idealGaitDict['trot'][limbtitle]
        bound = idealGaitDict['bound'][limbtitle] 
        for im, m in enumerate(mice):
            df_sub = df[df['mouseID'] == m]
            df_grouped = df_sub.groupby(pd.cut(df_sub[param], param_split)) 
            group_row_ids = df_grouped.groups
            grouped_dict = {key:df_sub.loc[val,limb].values for key,val in group_row_ids.items()} 
            keys = list(grouped_dict.keys())
            for i, gkey in enumerate(keys):
                values = grouped_dict[gkey][~np.isnan(grouped_dict[gkey])]
                if values.shape[0] < (Config.passiveOpto_config["stride_num_threshold"]):
                    print(f"Mouse {m} data excluded from group {gkey} because there are only {values.shape[0]} valid trials!")
                    continue
                
                trot_cots[im, i, il] = utils_math.circular_optimal_transport(trot*2*np.pi, grouped_dict[gkey]*2*np.pi)
                bound_cots[im, i, il] = utils_math.circular_optimal_transport(bound*2*np.pi, grouped_dict[gkey]*2*np.pi)
                
    for ig, (gait_cots, lnst) in enumerate(zip([trot_cots, bound_cots],
                                             ['solid', 'dotted'])):       
        sem = scipy.stats.sem(np.nanmean(gait_cots, axis = 2), axis = 0, nan_policy = 'omit') 
        ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., gait_cots.shape[0]-1)   
        ax.fill_between(np.arange(group_num), 
                        np.nanmean(np.nanmean(gait_cots, axis = 2), axis = 0) - ci, 
                        np.nanmean(np.nanmean(gait_cots, axis = 2), axis = 0) + ci, 
                        facecolor = FigConfig.colour_config[clr][clr_id], alpha = 0.2)
        ax.plot(np.arange(group_num),  
                np.nanmean(np.nanmean(gait_cots, axis = 2), axis = 0), 
                color = FigConfig.colour_config[clr][clr_id], 
                linestyle = lnst, 
                linewidth = 1)
        legend_colours[ix].append(FigConfig.colour_config[clr][clr_id])
        legend_linestyles[ix].append(lnst) 
    
    t, p = scipy.stats.ttest_ind(np.nanmean(trot_cots, axis = 2),
                                 np.nanmean(bound_cots, axis = 2),
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

ax.set_ylim(-0.1,0.4)        
ax.set_ylabel("Circular optimal transport")
ax.set_xlim(-0.5,group_num-1)
ax.set_xlabel('Speed (cm/s)')
xticklabels = [f'({k.left:.0f},{k.right:.0f}]' for k in group_row_ids.keys()]
ax.set_xticks(np.arange(group_num))
ax.set_xticklabels(xticklabels, rotation = 20, ha = 'right')

lgd_actors = [(legend_colours[i], legend_linestyles[i]) for i in range(ix+1)]
        
lgd = fig.legend(lgd_actors,
                ['head-fixed head height', 'head-fixed incline',
                 'head-free level', 'head-free incline'],
                handler_map={tuple: AnyObjectHandlerDouble()}, loc = 'upper left',
                bbox_to_anchor=(0,-0.52,1,0.3), mode="expand", borderaxespad=0.1,
                ncol = 1) 

lgd_actors2 = [(list(np.unique(legend_colours)), np.unique(legend_linestyles)[i]) for i in range(len(np.unique(legend_linestyles)))]

lgd2 = fig.legend([(lgd_actors2[0][0], "dotted"), (lgd_actors2[1][0], "solid")],
                ['bound-like', 'trot-like'],
                handler_map={tuple: AnyObjectHandler()}, loc = 'upper left',
                bbox_to_anchor=(0.3,0.75,0.7,0.2), mode="expand", borderaxespad=0.1,
                ncol = 1, frameon = False) 

figtitle = f"{yyyymmdd}_idealised_comparison_limbAverage.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight')