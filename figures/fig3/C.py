import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math
import os
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader, utils_processing
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandlerDouble

limb = 'lF0'
refLimb = 'lH1'
datafrac = 0.5
samples = 10000
inter = 'TRUE' 
param_type = 'continuous'
ylim = [140,180]

legend_colours = [[],[],[],[],[]]
legend_linestyles = [[],[],[],[],[]]
legend_labels = ['[0,0.4π]','(0.4π,0.8π]','(0.8π,1.2π]','(1.2π,1.6π]','(1.6π,2π]']

fig, ax = plt.subplots(1,1, figsize = (1.2,1.4)) #1.4,1.4
for x, (yyyymmdd, mouse_str, clr_str, lnst, pred, stat_dist) in enumerate(zip(['2021-10-23', '2022-05-06'],
                                                 ['mice_level', 'mice_incline'],
                                                 ['greys', 'homolateral'],
                                                 ['dashed', 'solid'],
                                                 ['speed', 'speed_incline'],
                                                 [[2,4,6], [84,86,88]])):
    modpath = Path(Config.paths["mtTreadmill_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_snoutBodyAngle_vs_{limb}_ref{refLimb}_{pred}_interaction{inter}_{param_type}_randMouse_samples{samples}.csv"
    mod = pd.read_csv(modpath)
    
    modpath_bootstrap = Path(Config.paths["mtTreadmill_output_folder"]) / f"{yyyymmdd}_mixedEffectsModelBOOTSTRAP_snoutBodyAngle_vs_{limb}_ref{refLimb}_{pred}_interaction{inter}_{param_type}_randMouse_samples{samples}.csv"
    mod_bootstrap = pd.read_csv(modpath_bootstrap)
    
    modpath_comps = Path(Config.paths["mtTreadmill_output_folder"]) / f"{yyyymmdd}_mixedEffectsModelCOMPS_snoutBodyAngle_vs_{limb}_ref{refLimb}_{pred}_interaction{inter}_{param_type}_randMouse_samples{samples}.csv"
    mod_comps = pd.read_csv(modpath_comps) 
    
    # dealing with the limb_cat phase variable
    limb_cat_predictors = [x for x in mod.iloc[:,0] if 'limb' in x]
    limb_group_num = int(len(limb_cat_predictors)/2 +1) 
    
    datafull = data_loader.load_processed_data(outputDir = Config.paths["mtTreadmill_output_folder"],
                                               dataToLoad = 'strideParams', 
                                               yyyymmdd = yyyymmdd, 
                                               limb = refLimb,
                                               appdx = '')[0]
    
    speedfull = utils_processing.remove_outliers(datafull['speed'])
    x_speed = np.linspace(5, int(np.max(speedfull)),endpoint = True)
    x_speed_centr = x_speed -np.nanmean(x_speed)
    
    angle_full = utils_processing.remove_outliers(datafull['snoutBodyAngle'])
    
    if 'incline' in pred:
        datafull['incline'] = [int(x[3:]) for x in datafull['trialType']]
        incline_full = utils_processing.remove_outliers(datafull['incline'])
        incline = np.nanmedian(datafull['incline'])- np.nanmean(datafull['incline'])
      
    unique_traces = np.empty((0))
    i_traces = np.empty((0))
    for i in range(limb_group_num):
        if i == 4:
            continue

        if i == 0:
            if 'incline' in pred:
                bootstrap_arr = np.asarray(mod_bootstrap.loc[:,'(Intercept)']).reshape(-1,1) + \
                                np.asarray(mod_bootstrap.loc[:,'pred1']).reshape(-1,1) @ x_speed_centr.reshape(-1,1).T + \
                                np.asarray(mod_bootstrap.loc[:,'pred2']).reshape(-1,1) * incline + \
                                np.nanmean(angle_full)
                y_angle = float(mod['Estimate'][mod['Unnamed: 0'] == '(Intercept)']) + \
                          float(mod['Estimate'][mod['Unnamed: 0'] == 'pred1'])*x_speed_centr + \
                          float(mod['Estimate'][mod['Unnamed: 0'] == 'pred2'])*incline + \
                          np.nanmean(angle_full)
            else:
                bootstrap_arr = np.asarray(mod_bootstrap.loc[:,'(Intercept)']).reshape(-1,1) + \
                                np.asarray(mod_bootstrap.loc[:,'pred1']).reshape(-1,1) @ x_speed_centr.reshape(-1,1).T + \
                                np.nanmean(angle_full)
                y_angle = float(mod['Estimate'][mod['Unnamed: 0'] == '(Intercept)']) + \
                          float(mod['Estimate'][mod['Unnamed: 0'] == 'pred1'])*x_speed_centr + \
                          np.nanmean(angle_full)
        
        else:
            if 'incline' in pred:
                bootstrap_arr = np.asarray(mod_bootstrap.loc[:,'(Intercept)']).reshape(-1,1) + \
                                np.asarray(mod_bootstrap.loc[:,'pred1']).reshape(-1,1) @ x_speed_centr.reshape(-1,1).T + \
                                np.asarray(mod_bootstrap.loc[:,'pred2']).reshape(-1,1) * incline + \
                                np.asarray(mod_bootstrap.loc[:, f'limb_cat{i+1}']).reshape(-1,1) + \
                                np.asarray(mod_bootstrap.loc[:, f'limb_cat{i+1}:pred1']).reshape(-1,1) @ x_speed_centr.reshape(-1,1).T + \
                                np.nanmean(angle_full)
                y_angle = float(mod['Estimate'][mod['Unnamed: 0'] == '(Intercept)']) + \
                          float(mod['Estimate'][mod['Unnamed: 0'] == 'pred1'])*x_speed_centr + \
                          float(mod['Estimate'][mod['Unnamed: 0'] == 'pred2'])*incline + \
                          float(mod['Estimate'][mod['Unnamed: 0'] == f'limb_cat{i+1}']) + \
                          float(mod['Estimate'][mod['Unnamed: 0'] == f'limb_cat{i+1}:pred1']) * x_speed_centr + \
                          np.nanmean(angle_full)
            else:
                bootstrap_arr = np.asarray(mod_bootstrap.loc[:,'(Intercept)']).reshape(-1,1) + \
                                np.asarray(mod_bootstrap.loc[:,'pred1']).reshape(-1,1) @ x_speed_centr.reshape(-1,1).T + \
                                np.asarray(mod_bootstrap.loc[:, f'limb_cat{i+1}']).reshape(-1,1) + \
                                np.asarray(mod_bootstrap.loc[:, f'limb_cat{i+1}:pred1']).reshape(-1,1) @ x_speed_centr.reshape(-1,1).T + \
                                np.nanmean(angle_full)
                y_angle = float(mod['Estimate'][mod['Unnamed: 0'] == '(Intercept)']) + \
                          float(mod['Estimate'][mod['Unnamed: 0'] == 'pred1'])*x_speed_centr + \
                          float(mod['Estimate'][mod['Unnamed: 0'] == f'limb_cat{i+1}']) + \
                          float(mod['Estimate'][mod['Unnamed: 0'] == f'limb_cat{i+1}:pred1']) * x_speed_centr + \
                          np.nanmean(angle_full)
    
        ax.fill_between(x_speed, 
                        np.percentile(bootstrap_arr, 2.5, axis = 0), 
                        np.percentile(bootstrap_arr, 97.5, axis = 0), 
                        facecolor = FigConfig.colour_config[clr_str][i], 
                        alpha = 0.3)

        ax.plot(x_speed, 
                y_angle, 
                color = FigConfig.colour_config[clr_str][i], 
                linewidth = 1,
                linestyle = lnst)
        
        unique_traces = np.append(unique_traces, round(y_angle[-1],6))
        i_traces = np.append(i_traces, i)
        
        legend_colours[i].append(FigConfig.colour_config[clr_str][i])
        legend_linestyles[i].append(lnst)
    
    p_thresholds = np.asarray(FigConfig.p_thresholds)/math.factorial(limb_group_num)
    
    unique_traces_sub = unique_traces[np.where((unique_traces > ylim[0]) & (unique_traces < ylim[1]))[0]]
    unique_traces_sub_sorting = np.argsort(unique_traces_sub)
    unique_traces_sub_sorted = unique_traces_sub[unique_traces_sub_sorting]
    i_traces_sorted = i_traces[unique_traces_sub_sorting].astype(int)
    for i, (y_tr, y_tr_id, i_tr) in enumerate(zip(unique_traces_sub_sorted, 
                                                  unique_traces_sub_sorting,
                                                  i_traces_sorted)):
        ax.hlines(y = y_tr, xmin = x_speed[-1]+stat_dist[0], xmax = x_speed[-1]+stat_dist[1], linewidth = 0.5, color = 'black', linestyle = lnst)
        if i < len(unique_traces_sub)-1:
            ax.vlines(x = x_speed[-1]+stat_dist[1], ymin = y_tr, ymax = unique_traces_sub_sorted[i+1], linewidth = 0.5, color = 'black', linestyle = lnst)
            y_tr_average = np.mean((y_tr, unique_traces_sub_sorted[i+1]))
            ax.hlines(y = y_tr_average, xmin = x_speed[-1]+stat_dist[1], xmax = x_speed[-1]+stat_dist[2], linewidth = 0.5, color = 'black', linestyle = lnst)
            ids = [i_traces_sorted[i], i_traces_sorted[i+1]]
            if 0 in ids:
                rowname = f"limb_cat{np.max(ids)+1}"
                pval = float(mod[mod['Unnamed: 0'] == rowname]['Pr(>|t|)'])
            else:
                rowname = f"limb_cat{np.min(ids)+1} vs limb_cat{np.max(ids)+1}"
                pval = float(mod_comps[mod_comps['Unnamed: 0'] == rowname]['Pr(>|z|)'])
            print(rowname, pval)

            ptext = ('*' * (pval < p_thresholds).sum())
            y_delta = -1.5
            if (pval < p_thresholds).sum() == 0:
                ptext = "n.s."
                y_delta = -1.2
            
            if x == 0 and i == 0:
                ax.text(x_speed[-1]+stat_dist[0] + 5, y_tr_average + y_delta - 1, ptext)
            else:
                ax.text(x_speed[-1]+stat_dist[0] + 5, y_tr_average + y_delta, ptext)       
        
ax.set_xlim(0,180)     
ax.set_xticks([0,60,120,180])
ax.set_xlabel('Speed (cm/s)')
ax.set_ylim(ylim[0],ylim[1])
ax.set_yticks([140,150,160,170,180])
ax.set_ylabel('Snout-hump angle (deg)')

# lgd = ax.legend([(legend_colours[0],legend_linestyles[0]), 
#                   (legend_colours[1],legend_linestyles[1]),
#                   (legend_colours[2],legend_linestyles[2]),
#                   (legend_colours[3],legend_linestyles[3])], 
#                 [legend_labels[0], legend_labels[1], legend_labels[2], legend_labels[3]],
#                 handler_map={tuple: AnyObjectHandlerDouble()}, loc = 'lower left',
#                 bbox_to_anchor=(-0.35,1.05,1.45,0.3), mode="expand", borderaxespad=0.1,
#                 title = "Homolateral phase", ncol = 2, handlelength = 1.4)

# 1-col legend for a narrow version of the panel
lgd = ax.legend([(legend_colours[0],legend_linestyles[0]), 
                  (legend_colours[1],legend_linestyles[1]),
                  (legend_colours[2],legend_linestyles[2]),
                  (legend_colours[3],legend_linestyles[3])], 
                [legend_labels[0], legend_labels[1], legend_labels[2], legend_labels[3]],
                handler_map={tuple: AnyObjectHandlerDouble()}, loc = 'lower left',
                bbox_to_anchor=(0.1,0.7,0.9,0.5), mode="expand", borderaxespad=0.1,
                title = "Homolateral phase", ncol = 1, handlelength = 1.4)

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'{yyyymmdd}_motorised_categorical_AVERAGED_snoutBodyAngle_v_speed.svg'),
            dpi = 300, 
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight')  
 