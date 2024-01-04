import sys
from pathlib import Path
import pandas as pd
import os
import numpy as np
import scipy.stats
import pickle
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from processing import data_loader, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandlerDouble

yyyymmdd = '2022-08-18'
refLimb = 'COMBINED'
limb = 'homolateral'
group_num = 5

legend_colours = np.empty((2, 0)).tolist()
legend_linestyles = np.empty((2, 0)).tolist()

fig, ax = plt.subplots(1,2,figsize = (1.5*2,1.5), sharey = True) # (1.7*2,1.7) for 4 columns
for axid, appdx, param, clrs, ploc in zip(
        [0,1,1],
        ['_incline', '', '_incline'],
        ['headLVL', 'snoutBodyAngle', 'snoutBodyAngle'],
        ['homolateral', 'greys7', 'homolateral'],
        [0.033, 0.033, 0.046]
        ):
    
    # load the combined rH1/lH1 ref dataset
    df, _, _ = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                                dataToLoad = "strideParams", 
                                                yyyymmdd = yyyymmdd,
                                                appdx = appdx,
                                                limb = refLimb)
    
    idealGaitDict = pickle.load(open(Path(Config.paths["passiveOpto_output_folder"]) / "idealisedGaitDict.pkl", "rb" ))
    trot = idealGaitDict['trot'][limb]
    tgallopR = idealGaitDict['transverse_gallop_R'][limb] #transverse gallop w limbRef entering stance first! "nonref-leading"
    
    if param == "headLVL":
        df[param] = [int(d[3:])*-1 for d in df[param]]
        data_split = np.linspace(df[param].min()-0.0000001, df[param].max(), group_num+1)   
    else:
        data_split = [141,147,154,161,167,174]
    
    trot_cots = np.empty((len(np.unique(df['mouseID'])), group_num))
    tgallopR_cots = np.empty((len(np.unique(df['mouseID'])), group_num))
    trot_cots[:] = np.nan
    tgallopR_cots[:] = np.nan
    
    for im, m in enumerate(np.unique(df['mouseID'])):
        df_sub = df[df['mouseID'] == m]
        df_grouped = df_sub.groupby(pd.cut(df_sub[param], data_split)) 
        group_row_ids = df_grouped.groups
        grouped_dict = {key:df_sub.loc[val,limb+'0'].values for key,val in group_row_ids.items()} 
        keys = list(grouped_dict.keys())
        for i, gkey in enumerate(keys):
            values = grouped_dict[gkey][~np.isnan(grouped_dict[gkey])]
            if values.shape[0] < (Config.passiveOpto_config["stride_num_threshold"]):
                print(f"Mouse {m} data excluded from group {gkey} because there are only {values.shape[0]} valid trials!")
                continue
            
            trot_cots[im, i] = utils_math.circular_optimal_transport(trot*2*np.pi, grouped_dict[gkey]*2*np.pi)
            tgallopR_cots[im, i] = utils_math.circular_optimal_transport(tgallopR*2*np.pi, grouped_dict[gkey]*2*np.pi)
     
    clr_id = 2 if clrs == 'homolateral' else 0
    for ig, (gait_cots, lnst) in enumerate(zip([trot_cots, tgallopR_cots],
                                             ['solid', 'dashed'])):       
        sem = scipy.stats.sem(gait_cots, axis = 0, nan_policy = 'omit')  
        ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., gait_cots.shape[0]-1)  
        ax[axid].fill_between(np.arange(group_num), np.nanmean(gait_cots, axis = 0) - ci, np.nanmean(gait_cots, axis = 0) + ci, facecolor = FigConfig.colour_config[clrs][clr_id], alpha = 0.2)
        ax[axid].plot(np.arange(group_num),  np.nanmean(gait_cots, axis = 0), color = FigConfig.colour_config[clrs][clr_id], linestyle = lnst, linewidth = 1)
        
        if param == 'snoutBodyAngle':
            legend_colours[ig].append( FigConfig.colour_config[clrs][clr_id])
            legend_linestyles[ig].append(lnst)
            
        
    t, p = scipy.stats.ttest_ind(trot_cots, tgallopR_cots, axis = 0, equal_var =False, nan_policy = 'omit')
    pvals = np.empty((group_num), dtype = 'object')
    for ip, pval in enumerate(p.data):
        p_text = ('*' * (pval < np.asarray(FigConfig.p_thresholds)).sum())
        if (pval < np.asarray(FigConfig.p_thresholds)).sum() == 0 and not np.isnan(pval):
            p_text = "n.s."
            ax[axid].text(ip, ploc+0.005, p_text, ha = 'center', color = FigConfig.colour_config[clrs][clr_id])
        else:
            ax[axid].text(ip, ploc, p_text, ha = 'center', color = FigConfig.colour_config[clrs][clr_id])
    
    ax[axid].set_yticks(np.linspace(0.04,0.28,7,endpoint = True))
    ax[axid].set_ylim(0.03,0.28)
    
    ax[axid].set_xlim(-0.5, group_num-1)
    ax[axid].set_xticks(np.linspace(0, group_num-1, group_num, endpoint =True))
    ax[axid].set_xticklabels([f'{np.mean([keys[j].left,keys[j].right]):.0f}' for j in range(group_num)])

ax[0].set_ylabel("Dissimilarity (a.u.)")
ax[0].set_xlabel('Incline (deg)')
ax[1].set_xlabel('Snout-hump angle (deg)')

lgd = fig.legend([(legend_colours[0],legend_linestyles[0]), 
                  (legend_colours[1],legend_linestyles[1])], 
                ['trot', 'transverse gallop'],
                handler_map={tuple: AnyObjectHandlerDouble()}, loc = 'lower left',
                bbox_to_anchor=(0.2,0.95,0.7,0.3), mode="expand", borderaxespad=0.1,
                title = "Idealised gait for comparison", ncol = 2)    

ax[1].hlines(0.275, xmin = 0.01, xmax = 0.6, linestyle = 'solid', color = FigConfig.colour_config['homolateral'][2], linewidth = 1)
ax[1].hlines(0.268, xmin = 0.01, xmax = 0.6, linestyle = 'dashed', color = FigConfig.colour_config['homolateral'][2], linewidth = 1)
ax[1].text(0.7, 0.266, "surface slope")
ax[1].hlines(0.256, xmin = 0.02, xmax = 0.6, linestyle = 'solid', color = FigConfig.colour_config['greys7'][0], linewidth = 1)
ax[1].hlines(0.249, xmin = 0.02, xmax = 0.6, linestyle = 'dashed', color = FigConfig.colour_config['greys7'][0], linewidth = 1)
ax[1].text(0.7, 0.247, "head height")
plt.tight_layout(w_pad = 4)    

figtitle = f"MS3_{yyyymmdd}_idealised_comparison.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight')