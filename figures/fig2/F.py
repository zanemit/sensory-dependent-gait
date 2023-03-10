import sys
from pathlib import Path
import pandas as pd
import os
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader, utils_processing, utils_math
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandlerDouble

yyyymmdd = '2022-08-18'
refLimb = 'lH1'
datafrac = 0.5
iterations = 1000

legend_colours = [[],[],[],[],[]]
legend_linestyles = [[],[],[],[],[]]
legend_labels = []

fig, ax = plt.subplots(3,1,figsize = (1.7,1.5*3), sharex = True) #1.4,1.5 for S1 refrH1
for axid, (limb, ylim, yticks, yticklabels, mod_samples, colours, stat_dists, y_tr_shifts, y_hline) in enumerate(zip(
                                            ['lF0','rF0','rH0'],
                                            [(0.3*np.pi,1.5*np.pi), (-0.5*np.pi,1.5*np.pi), (-np.pi,np.pi)],
                                            [[0.5*np.pi,np.pi,1.5*np.pi], [-0.5*np.pi,0,0.5*np.pi,np.pi,1.5*np.pi], [-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi] ],
                                            [["0.5π", "π", "1.5π"] , ["-0.5π","0", "0.5π", "π", "1.5π"] , ["-π","-0.5π","0", "0.5π", "π"]  ],
                                            [[9075, 8125],[8483, 7629],[9120, 7959]],
                                            [['greys', 'homolateral'], ['greys', 'diagonal'], ['greys', 'homologous']],
                                            [[[28,44,46], [2,4,6]], [[32,48,50], [2,4,6]], [[30,46,48], [2,4,6]]],
                                            [[[0,0,0,0],[0,0,0,0]] , [[-0.45,-0.25,-0.1,0.2],[0,0.3,0.6]], [[-0.5,-0.25,0,0.25],[0,0.25,0.45]]],
                                            [[1.37,1.26], [1.37,1.2], [0.82,0.6]]
                                            )):
    for w, (predictors, sBA_split, appdx, lnst) in enumerate(zip(
            [['speed', 'snoutBodyAngle'],['speed', 'snoutBodyAngle', 'incline']],
            [[141,147,154,161,167,174],[147,154,161,167,174]],
            ['', '_incline'],
            ['dashed', 'solid']
            )):

        samples = mod_samples[w]
        clrs = colours[w]
        stat_dist = stat_dists[w] # x1: distance to trace, x2-x1: first hline length, x3-x2 second hline length
        y_tr_shift = y_tr_shifts[w]
        
        sBA_split_str = '-'.join(np.asarray(sBA_split).astype(str))
        
        if len(predictors) == 2:
            beta1path = Path(Config.paths["passiveOpto_output_folder"]) / f"{yyyymmdd}_beta1_{limb}_ref{refLimb}_{predictors[0]}_{predictors[1]}_interactionFALSE_categorical_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
            beta2path = Path(Config.paths["passiveOpto_output_folder"]) / f"{yyyymmdd}_beta2_{limb}_ref{refLimb}_{predictors[0]}_{predictors[1]}_interactionFALSE_categorical_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
            c = 0
        else:
            beta1path = Path(Config.paths["passiveOpto_output_folder"]) / f"{yyyymmdd}_beta1_{limb}_ref{refLimb}_{predictors[0]}_{predictors[1]}_{predictors[2]}_interactionFALSE_categorical_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
            beta2path = Path(Config.paths["passiveOpto_output_folder"]) / f"{yyyymmdd}_beta2_{limb}_ref{refLimb}_{predictors[0]}_{predictors[1]}_{predictors[2]}_interactionFALSE_categorical_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
            c = 1
            
        beta1 = pd.read_csv(beta1path)
        beta2 = pd.read_csv(beta2path)
    
        datafull = data_loader.load_processed_data(dataToLoad = 'strideParams', 
                                                   yyyymmdd = yyyymmdd, 
                                                   limb = refLimb, 
                                                   appdx = appdx)[0]
        datafull_relevant = utils_processing.remove_outliers(datafull['snoutBodyAngle'])
        speedfull = utils_processing.remove_outliers(datafull['speed'])
        x_speed = np.arange(5, int(np.max(speedfull)),1)
        
        pred3_num = len([k for k in beta1.columns if 'pred2' in k]) + 1
        phase3_preds = np.empty((beta1.shape[0], x_speed.shape[0],pred3_num))
        phase3_preds[:] = np.nan
        for i in range(pred3_num):
            if i == 0:
                mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) @ x_speed.reshape(-1,1).T
                mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) @ x_speed.reshape(-1,1).T
                phase3_preds[:, :, i] = np.arctan2(mu2, mu1)
    
            else:
                mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) @ x_speed.reshape(-1,1).T +np.asarray(beta1[f'pred2{i+1}']).reshape(-1,1)
                mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) @ x_speed.reshape(-1,1).T +np.asarray(beta2[f'pred2{i+1}']).reshape(-1,1)
                phase3_preds[:, :, i] = np.arctan2(mu2, mu1)
    
        unique_traces = np.empty(())
        for i in range(pred3_num):
            print(f"Working on group {i}...")
            # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
            for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
                print(f"Working on data range {k}...")
                if k == 1:
                    phase3_preds[phase3_preds<0] = phase3_preds[phase3_preds<0]+2*np.pi
                if k == 2:
                    phase3_preds[phase3_preds<np.pi] = phase3_preds[phase3_preds<np.pi]+2*np.pi
                    if not (w == 0 and axid>0):
                        legend_colours[i+c].append(FigConfig.colour_config[clrs][i+c])
                        legend_linestyles[i+c].append(lnst)
                    
                    if len(predictors) == 2:
                        legend_labels.append(f"({sBA_split[i]},{sBA_split[i+1]}]")
                trace = scipy.stats.circmean(phase3_preds[:,:,i],high = hi, low = lo, axis = 0)
                lower = np.zeros_like(trace)
                higher = np.zeros_like(trace)
                for x in range(lower.shape[0]):
                    lower[x], higher[x] =  utils_math.hpd_circular(phase3_preds[:,x,i], mass_frac = 0.95, high = hi, low = lo)# % (np.pi*2)
                
                if round(trace[-1],6) not in unique_traces and not np.any(abs(np.diff(trace))>5):
                    unique_traces = np.append(unique_traces, round(trace[-1],6))
                    print('plotting...')
                    ax[axid].fill_between(x_speed, lower, higher, alpha = 0.2, facecolor = FigConfig.colour_config[clrs][i+c])
                    ax[axid].plot(x_speed, trace, color = FigConfig.colour_config[clrs][i+c], linewidth = 1, linestyle = lnst)
    
                    
        # statistics
        if len(predictors) == 2:
            statspath = Path(Config.paths["passiveOpto_output_folder"]) / f"{yyyymmdd}_coefCircCategorical_{limb}_ref{refLimb}_{predictors[0]}_{predictors[1]}_interactionFALSE_categorical_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"    
        else:
            statspath = Path(Config.paths["passiveOpto_output_folder"]) / f"{yyyymmdd}_coefCircCategorical_{limb}_ref{refLimb}_{predictors[0]}_{predictors[1]}_{predictors[2]}_interactionFALSE_categorical_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        stats = pd.read_csv(statspath)
        
        unique_traces_sub = unique_traces[np.where((unique_traces > ylim[0]) & (unique_traces < ylim[1]))[0]]
        unique_traces_sub_sorting = np.argsort(unique_traces_sub)
        unique_traces_sub_sorted = unique_traces_sub[unique_traces_sub_sorting]
        for i, (y_tr, y_tr_id) in enumerate(zip(unique_traces_sub_sorted, unique_traces_sub_sorting)):
            ax[axid].hlines(y = y_tr, xmin = x_speed[-1]+stat_dist[0], xmax = x_speed[-1]+stat_dist[1], linewidth = 0.5, color = 'black', linestyle = lnst)
            if i < len(unique_traces_sub)-1:
                ax[axid].vlines(x = x_speed[-1]+stat_dist[1], ymin = y_tr, ymax = unique_traces_sub_sorted[i+1], linewidth = 0.5, color = 'black', linestyle = lnst)
                y_tr_average = np.mean((y_tr, unique_traces_sub_sorted[i+1]))
                ax[axid].hlines(y = y_tr_average, xmin = x_speed[-1]+stat_dist[1], xmax = x_speed[-1]+stat_dist[2], linewidth = 0.5, color = 'black', linestyle = lnst)
                ids = [y_tr_id, unique_traces_sub_sorting[i+1]]
                if 0 in ids:
                    rowname = f"pred2{np.max(ids)+1}"
                else:
                    rowname = f"pred2{np.min(ids)+1}pred2{np.max(ids)+1}"
                sign = np.sign((stats[stats['Unnamed: 0'] == rowname]['LB'],stats[stats['Unnamed: 0'] == rowname]['UB'])).sum()
                if sign == 0:
                    ptext = "n.s."
                else:
                    ptext = "*"
                    y_tr_average -= 0.12
                ax[axid].text(x_speed[-1]+stat_dist[2]+2, y_tr_average + y_tr_shift[i], ptext)
     
    # axes 
    ax[axid].set_ylim(ylim[0], ylim[1])
    ax[axid].set_yticks(yticks)
    ax[axid].set_yticklabels(yticklabels)
    ax[axid].set_ylabel(f'Predicted {limb[:2]} phase (rad)')
    
    const = 3
    if axid == 0:
        const = 2
    ax[axid].hlines((y_hline[0]+(0.03*const))*np.pi, xmin = 20, xmax = 50, linestyle = 'solid', color = FigConfig.colour_config[colours[1]][1], linewidth = 1)
    ax[axid].hlines((y_hline[0]+(0.015*const))*np.pi, xmin = 20, xmax = 50, linestyle = 'solid', color = FigConfig.colour_config[colours[1]][3], linewidth = 1)
    ax[axid].text(55, y_hline[0]*np.pi, "incline trials")
    ax[axid].hlines((y_hline[1]+(0.03*const))*np.pi, xmin = 20, xmax = 50, linestyle = 'dashed', color = FigConfig.colour_config[colours[0]][1], linewidth = 1)
    ax[axid].hlines((y_hline[1]+(0.015*const))*np.pi, xmin = 20, xmax = 50, linestyle = 'dashed', color = FigConfig.colour_config[colours[0]][3], linewidth = 1)
    ax[axid].text(55,  y_hline[1]*np.pi, "head height trials")


ax[axid].set_xticks([0,50,100,150,200])
ax[axid].set_xlim(0,200)
ax[axid].set_xlabel('Speed (cm/s)')

fig.tight_layout(h_pad = 3)
lgd = ax[0].legend([(legend_colours[0],legend_linestyles[0]), 
                 (legend_colours[1],legend_linestyles[1]),
                 (legend_colours[2],legend_linestyles[2]),
                 (legend_colours[3],legend_linestyles[3]),
                 (legend_colours[4],legend_linestyles[4])], 
                [legend_labels[0], legend_labels[1], legend_labels[2], legend_labels[3], legend_labels[4]],
                handler_map={tuple: AnyObjectHandlerDouble()}, loc = 'lower left',
                bbox_to_anchor=(-0.2,1.05,1.3,0.3), mode="expand", borderaxespad=0.1,
                title = "Snout-hump angle (deg)", ncol = 2, handlelength = 1.4)

figtitle = ('_').join(np.concatenate(([yyyymmdd], [f'ALL_ref{refLimb}'], ['speed','snoutBodyAngle'],str(beta1path).split('_')[-8:-2]))) + '.svg'
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), dpi = 300, bbox_extra_artists = (lgd, ), bbox_inches = 'tight')
