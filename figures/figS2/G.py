import sys
from pathlib import Path
import pandas as pd
import os
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_processing, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandlerDouble

yyyymmdd = '2022-08-18'
outputDir = Config.paths['passiveOpto_output_folder']
datafrac = 0.3
iterations = 1000
ref = 'COMBINED'
limb = 'homolateral0'

appdx_dict = {2: '', 3: '_incline'}
sample_nums = {'_incline': 9280, '': 10453}

ylim = (0.3*np.pi,1.5*np.pi)
yticks = [0.5*np.pi,np.pi,1.5*np.pi]
yticklabels = ["0.5π", "π", "1.5π"]  

legend_colours = [[],[]]
legend_linestyles = [[],[]]

fig, ax = plt.subplots(1,1,figsize = (1.7,1.6), sharey = True) #1.6,1.4 for 4figs S2 bottom row

predictors = ['speed', 'snoutBodyAngle', 'incline']
param_col = 'snoutBodyAngle'
interaction = 'TRUEsecondary'
clrs = 'homolateral'
tlt = 'Slope trials'
    
sBA_split_str = 'FALSE'
    
appdx = appdx_dict[len(predictors)]
samples = sample_nums[appdx]
      
beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
statspath = Path(outputDir) / f"{yyyymmdd}_coefCircCategorical_homolateral0_refCOMBINED_{predictors[0]}_{predictors[1]}_{predictors[2]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"

beta1 = pd.read_csv(beta1path)
beta2 = pd.read_csv(beta2path)
stats = pd.read_csv(statspath)

datafull = data_loader.load_processed_data(dataToLoad = 'strideParams', 
                                           yyyymmdd = yyyymmdd,
                                           limb = ref, 
                                           appdx = appdx)[0]

if 'deg' in datafull['headLVL'][0]:
    datafull['incline'] = [-int(x[3:]) for x in datafull['headLVL']]
    
predictor = 'incline'
pred = 'pred3' #incline
nonpred = 'pred2'
xlim = (-41,45)
ax.set_xlim(xlim[0], xlim[1])
ax.set_xticks([-40,-20,0,20,40])
ax.set_xlabel('Incline (deg)')

y_tr_list = np.empty((0)) # 5 percentiles to plot = 5 pairs to compare!
x_tr_list = np.empty((0))
p_tr_list = np.empty((0))
y_delta_list = np.empty((0))
    
# define the range of snout-hump angles or inclines (x axis predictor)
pred2_relevant = utils_processing.remove_outliers(datafull[predictor]) 
pred2_centred = pred2_relevant - np.nanmean(pred2_relevant) #centering
pred2_range = np.linspace(pred2_centred.min(), pred2_centred.max(), num = 100)
    
prcnt = 50
# find median param (excluding outliers)    
param_relevant = utils_processing.remove_outliers(datafull[param_col])
param = np.percentile(param_relevant, prcnt) - np.nanmean(param_relevant)

# initialise arrays
phase2_preds = np.empty((beta1.shape[0], pred2_range.shape[0]))
phase2_preds[:] = np.nan

speed_relevant = utils_processing.remove_outliers(datafull['speed'])
speed = np.percentile(speed_relevant, 50) - np.nanmean(speed_relevant)

unique_traces = np.empty((0))
for i, (refLimb, lnst) in enumerate(zip(['', 'rH1'],
                                        ['solid', 'dashed']
                                            )):
        
    if refLimb == 'rH1':
        mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) * speed + np.asarray(beta1[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1[nonpred]).reshape(-1,1) * param + np.asarray(beta1["pred2:pred3"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) + np.asarray(beta1[f"pred4{refLimb}"]).reshape(-1,1) #the mean of the third predictor (if present) is zero because it was centred before modelling
        mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) * speed + np.asarray(beta2[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta2[nonpred]).reshape(-1,1) * param + np.asarray(beta2["pred2:pred3"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) + np.asarray(beta2[f"pred4{refLimb}"]).reshape(-1,1) 
    else:
        mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) * speed + np.asarray(beta1[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1[nonpred]).reshape(-1,1) * param + np.asarray(beta1["pred2:pred3"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) #the mean of the third predictor (if present) is zero because it was centred before modelling
        mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) * speed + np.asarray(beta2[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1[nonpred]).reshape(-1,1) * param + np.asarray(beta2["pred2:pred3"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) 
    phase2_preds[:,:] = np.arctan2(mu2, mu1)
    
    # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
    for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
        print(f"{refLimb}: working on data range {k}...")
        if k == 1:
            phase2_preds[phase2_preds<0] = phase2_preds[phase2_preds<0]+2*np.pi
        if k == 2:
            phase2_preds[phase2_preds<np.pi] = phase2_preds[phase2_preds<np.pi]+2*np.pi
            legend_colours[i].append(FigConfig.colour_config[clrs][2])
            legend_linestyles[i].append(lnst)
        
        trace = scipy.stats.circmean(phase2_preds[:,:],high = hi, low = lo, axis = 0)
        lower = np.zeros_like(trace); higher = np.zeros_like(trace)
        for x in range(lower.shape[0]):
            lower[x], higher[x] =  utils_math.hpd_circular(phase2_preds[:,x], 
                                                            mass_frac = 0.95, 
                                                            high = hi, 
                                                            low = lo) #% (np.pi*2)
        
        if round(trace[-1],6) not in unique_traces and not np.any(abs(np.diff(trace))>5):
            unique_traces = np.append(unique_traces, round(trace[-1],6))
            print('plotting...')    
            ax.fill_between(pred2_range + np.nanmean(pred2_relevant), 
                                  lower, 
                                  higher, 
                                  alpha = 0.2, 
                                  facecolor = FigConfig.colour_config[clrs][2]
                                    )
            ax.plot((pred2_range + np.nanmean(pred2_relevant)), 
                    trace, 
                    color = FigConfig.colour_config[clrs][2], 
                    linewidth = 1, 
                    linestyle = lnst, 
                    )
        
#statistics 
unique_traces_sub = unique_traces[np.where((unique_traces > ylim[0]) & (unique_traces < ylim[1]))[0]]
unique_traces_sub_sorting = np.argsort(unique_traces_sub)
unique_traces_sub_sorted = unique_traces_sub[unique_traces_sub_sorting]
for i, (y_tr, y_tr_id) in enumerate(zip(unique_traces_sub_sorted, unique_traces_sub_sorting)):
    ax.hlines(y = y_tr, 
                    xmin = np.nanmax(pred2_relevant) + 2, 
                    xmax = np.nanmax(pred2_relevant) + 4, 
                    linewidth = 0.5, 
                    color = 'black', 
                    linestyle = lnst)
    if i < len(unique_traces_sub)-1:
        ax.vlines(x = np.nanmax(pred2_relevant) + 4, 
                        ymin = y_tr, 
                        ymax = unique_traces_sub_sorted[i+1], 
                        linewidth = 0.5, 
                        color = 'black', 
                        linestyle = lnst)
        y_tr_average = np.mean((y_tr, unique_traces_sub_sorted[i+1]))
        sign = np.sign((stats['LB'],stats['UB'])).sum()
        if sign == 0:
            ptext = "n.s."
            y_delta = -0.02*np.pi
        else:
            ptext = "*"
            y_tr_average -= 0.12 # the * looks too high up otherwise
            y_delta = 0

        y_delta_list = np.append(y_delta_list, y_delta)
        y_tr_list = np.append(y_tr_list, y_tr_average)
        x_tr_list = np.append(x_tr_list, np.nanmax(pred2_relevant)+6)
        p_tr_list = np.append(p_tr_list, ptext)
        
# ax.text(np.mean(xlim)+((xlim[1]-xlim[0])*0.75/4)*(iprcnt-2), 1.3*np.pi, f"{prcnt}th", ha = 'center', color = FigConfig.colour_config[clrs][iprcnt])
prcnts_d = [0]  
y_tr_spread = np.median(y_tr_list) + prcnts_d # empirically determined that 0.2 is sufficient separation on my axis
for isp in range(len(y_tr_spread)):
    ax.text(x_tr_list[isp], 
                  y_tr_spread[isp],# + y_delta_list[isp], 
                  p_tr_list[isp])

# ax.text(np.mean(xlim), 1.37*np.pi, pct_text, color = FigConfig.colour_config[clrs][0], ha = 'center')

# axes 
ax.set_ylim(ylim[0], ylim[1])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_title(tlt)#, pad = 10)
    
ax.set_ylabel('Homolateral phase (rad)')

lgd = fig.legend([(legend_colours[0],legend_linestyles[0]), 
                  (legend_colours[1],legend_linestyles[1])], 
                ['left hind ref', 'right hind ref'],
                handler_map={tuple: AnyObjectHandlerDouble()}, loc = 'lower left',
                # bbox_to_anchor=(0.25,0.95,0.65,0.3), 
                bbox_to_anchor=(0.3,0.7,0.6,0.3), 
                mode="expand", borderaxespad=0.1,
                # title = "Reference limb", 
                ncol = 1)

plt.tight_layout()

figtitle = f"MS3_{yyyymmdd}_homolateral_incline_reflimbs.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight')
