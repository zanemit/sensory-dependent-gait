import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader, utils_processing, utils_math
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig

yyyymmdd = '2022-08-18'
predictors = ['speed', 'snoutBodyAngle', 'incline']
limb = 'lF0'
refLimb = 'lH1'
sBA_split = [147,154,161,167,174]
sBA_split_str = '-'.join(np.asarray(sBA_split).astype(str))
samples = 8125
datafrac = 0.5
iterations = 1000
appdx = "_incline"
stat_dist = [2,5,9]

beta1path = Path(Config.paths["passiveOpto_output_folder"]) / f"{yyyymmdd}_beta1_{limb}_ref{refLimb}_{predictors[0]}_{predictors[1]}_{predictors[2]}_interactionFALSE_categorical_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
beta2path = Path(Config.paths["passiveOpto_output_folder"]) / f"{yyyymmdd}_beta2_{limb}_ref{refLimb}_{predictors[0]}_{predictors[1]}_{predictors[2]}_interactionFALSE_categorical_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
inclines = [-40,-24,-8,8,24,40]

beta1 = pd.read_csv(beta1path)
beta2 = pd.read_csv(beta2path)

datafull = data_loader.load_processed_data(dataToLoad = 'strideParams', 
                                           yyyymmdd = yyyymmdd, 
                                           limb = refLimb, 
                                           appdx = appdx)[0]
datafull_relevant = utils_processing.remove_outliers(datafull['snoutBodyAngle'])
speedfull = utils_processing.remove_outliers(datafull['speed'])
x_speed = np.arange(5, int(np.max(speedfull)),1)

dataLabelDict = {'lF0': 'homolateral', 'rF0': 'diagonal', 'rH0': 'homologous'}
lnst = 'solid'

ylim = (0.5*np.pi,1.5*np.pi)
yticks = [0.5*np.pi,np.pi,1.5*np.pi]
yticklabels = ["0.5π", "π", "1.5π"]        
  
pred3_num = len([k for k in beta1.columns if 'pred3' in k]) + 1
phase3_preds = np.empty((beta1.shape[0], x_speed.shape[0],pred3_num))
phase3_preds[:] = np.nan
for i in range(pred3_num):
    if i == 0:
        mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) @ x_speed.reshape(-1,1).T
        mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) @ x_speed.reshape(-1,1).T
        phase3_preds[:, :, i] = np.arctan2(mu2, mu1)

    else:
        mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) @ x_speed.reshape(-1,1).T +np.asarray(beta1[f'pred3{i+1}']).reshape(-1,1)
        mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) @ x_speed.reshape(-1,1).T +np.asarray(beta2[f'pred3{i+1}']).reshape(-1,1)
        phase3_preds[:, :, i] = np.arctan2(mu2, mu1)

unique_traces = np.empty(())
fig, ax = plt.subplots(1,1,figsize = (1.4,1.4)) #1.4,1.5 for S1 refrH1
for i in range(pred3_num):
    print(f"Working on group {i}...")
    # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
    for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
        print(f"Working on data range {k}...")
        if k == 1:
            phase3_preds[phase3_preds<0] = phase3_preds[phase3_preds<0]+2*np.pi
        if k == 2:
            phase3_preds[phase3_preds<np.pi] = phase3_preds[phase3_preds<np.pi]+2*np.pi
        trace = scipy.stats.circmean(phase3_preds[:,:,i],high = hi, low = lo, axis = 0)
        lower = np.zeros_like(trace)
        higher = np.zeros_like(trace)
        for x in range(lower.shape[0]):
            lower[x], higher[x] =  utils_math.hpd_circular(phase3_preds[:,x,i], mass_frac = 0.95, high = hi, low = lo)# % (np.pi*2)
        
        if round(trace[-1],6) not in unique_traces and not np.any(abs(np.diff(trace))>5):
            unique_traces = np.append(unique_traces, round(trace[-1],6))
            print('plotting...')
            ax.fill_between(x_speed, lower, higher, alpha = 0.2, facecolor = FigConfig.colour_config[dataLabelDict[limb]][i])
            if i == 0:
                ax.plot(x_speed, trace, color = FigConfig.colour_config[dataLabelDict[limb]][i], linewidth = 1, linestyle = 'solid', label = f"[{inclines[i]},{inclines[i+1]}]")
            else:
                ax.plot(x_speed, trace, color = FigConfig.colour_config[dataLabelDict[limb]][i], linewidth = 1, linestyle = 'solid', label = f"({inclines[i]},{inclines[i+1]}]")

ax.set_ylim(ylim[0], ylim[1])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

# statistics
statspath = Path(Config.paths["passiveOpto_output_folder"]) / f"{yyyymmdd}_coefCircCategorical_{limb}_ref{refLimb}_{predictors[0]}_{predictors[1]}_{predictors[2]}_interactionFALSE_categorical_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
stats = pd.read_csv(statspath)

unique_traces_sub = unique_traces[np.where((unique_traces > ylim[0]) & (unique_traces < ylim[1]))[0]]
unique_traces_sub_sorting = np.argsort(unique_traces_sub)
unique_traces_sub_sorted = unique_traces_sub[unique_traces_sub_sorting]
for i, (y_tr, y_tr_id) in enumerate(zip(unique_traces_sub_sorted, unique_traces_sub_sorting)):
    ax.hlines(y = y_tr, xmin = x_speed[-1]+stat_dist[0], xmax = x_speed[-1]+stat_dist[1], linewidth = 0.5, color = 'black')
    if i < len(unique_traces_sub)-1:
        ax.vlines(x = x_speed[-1]+stat_dist[1], ymin = y_tr, ymax = unique_traces_sub_sorted[i+1], linewidth = 0.5, color = 'black')
        y_tr_average = np.mean((y_tr, unique_traces_sub_sorted[i+1]))
        ax.hlines(y = y_tr_average, xmin = x_speed[-1]+stat_dist[1], xmax = x_speed[-1]+stat_dist[2], linewidth = 0.5, color = 'black')
        ids = [y_tr_id, unique_traces_sub_sorting[i+1]]
        if 0 in ids:
            rowname = f"pred3{np.max(ids)+1}"
        else:
            rowname = f"pred3{np.min(ids)+1}pred3{np.max(ids)+1}"
        sign = np.sign((stats[stats['Unnamed: 0'] == rowname]['LB'],stats[stats['Unnamed: 0'] == rowname]['UB'])).sum()
        if sign == 0:
            ptext = "n.s."
        else:
            ptext = "*"
            y_tr_average -= 0.12
        ax.text(x_speed[-1]+stat_dist[2]+2, y_tr_average, ptext)
        
ax.set_xticks([0,50,100,150,200])
ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Predicted lF phase (rad)')

# legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))

lgd = fig.legend(handles = by_label.values(), labels = by_label.keys(), 
                  title = "Treadmill incline (deg)",
                loc = 'lower left', bbox_to_anchor=(0,0.92,0.9,0.3), 
                mode="expand", borderaxespad=0, ncol = 2, handlelength = 1.4)

figtitle = ('_').join(np.concatenate(([yyyymmdd], str(beta1path).split('_')[2:4], ['speed','snoutBodyAngle','incline'],str(beta1path).split('_')[-8:-2]))) + '.svg'
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), dpi = 300, bbox_extra_artists = (lgd, ), bbox_inches = 'tight')
