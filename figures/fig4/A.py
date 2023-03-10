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

fig, ax = plt.subplots(1,3,figsize = (1.65*3,1.6), sharey = True) #1.6,1.4 for 4figs S2 bottom row
for axid, (predictors, param_col, interaction, clrs, tlt, pct_text) in enumerate(zip(

        [['speed', 'snoutBodyAngle'], ['speed', 'snoutBodyAngle', 'incline'], ['speed', 'snoutBodyAngle', 'incline']],
        ['speed', 'snoutBodyAngle', 'incline'],
        ['TRUE', 'TRUEsecondary', 'TRUEsecondary'],
        ['greys', 'homolateral', 'homolateral'],
        ['Head height trials', 'Incline trials', 'Incline trials'],
        ['speed percentile', 'body tilt percentile', 'incline percentile']
        )):
    
    sBA_split_str = 'FALSE'
        
    appdx = appdx_dict[len(predictors)]
    samples = sample_nums[appdx]
          
    if len(predictors) > 2:
        beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        statspath = Path(outputDir) / f"{yyyymmdd}_coefCircCategorical_homolateral0_refCOMBINED_{predictors[0]}_{predictors[1]}_{predictors[2]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"

    else: #head height trials
        beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        statspath = Path(outputDir) / f"{yyyymmdd}_coefCircCategorical_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        
    beta1 = pd.read_csv(beta1path)
    beta2 = pd.read_csv(beta2path)
    stats = pd.read_csv(statspath)

    datafull = data_loader.load_processed_data(dataToLoad = 'strideParams', 
                                               yyyymmdd = yyyymmdd,
                                               limb = ref, 
                                               appdx = appdx)[0]
    
    if 'deg' in datafull['headLVL'][0]:
        datafull['incline'] = [-int(x[3:]) for x in datafull['headLVL']]
        
    if len(predictors)>2 and param_col == 'snoutBodyAngle' : # incline
        predictor = 'incline'
        pred = 'pred3' #incline
        nonpred = 'pred2'
        xlim = (-41,45)
        ax[axid].set_xlim(xlim[0], xlim[1])
        ax[axid].set_xticks([-40,-20,0,20,40])
        ax[axid].set_xlabel('Incline (deg)')
    else:
        predictor = 'snoutBodyAngle'
        pred = 'pred2'
        nonpred = 'pred3' #not used in head height trials
        xlim = (140,180)
        ax[axid].set_xlim(xlim[0],xlim[1])
        ax[axid].set_xticks([140,150,160,170,180])
        ax[axid].set_xlabel('Snout-hump angle (deg)')
    
    y_tr_list = np.empty((0)) # 5 percentiles to plot = 5 pairs to compare!
    x_tr_list = np.empty((0))
    p_tr_list = np.empty((0))
    y_delta_list = np.empty((0))
    
    if param_col != 'speed':
        # find median speed for incline trials
        speed_relevant = utils_processing.remove_outliers(datafull['speed'])
        speed = np.percentile(speed_relevant, 50) - np.nanmean(speed_relevant)
        
    # define the range of snout-hump angles or inclines (x axis predictor)
    pred2_relevant = utils_processing.remove_outliers(datafull[predictor]) 
    pred2_centred = pred2_relevant - np.nanmean(pred2_relevant) #centering
    pred2_range = np.linspace(pred2_centred.min(), pred2_centred.max(), num = 100)
        
    for iprcnt, prcnt in enumerate([5,25,50,75,95]):
        # find median param (excluding outliers)    
        param_relevant = utils_processing.remove_outliers(datafull[param_col])
        param = np.percentile(param_relevant, prcnt) - np.nanmean(param_relevant)
        
        # initialise arrays
        phase2_preds = np.empty((beta1.shape[0], pred2_range.shape[0]))
        phase2_preds[:] = np.nan
        
        unique_traces = np.empty((0))
        for i, (refLimb, lnst) in enumerate(zip(['', 'rH1'],
                                                ['solid', 'dashed']
                                                    )):
        
            if len(predictors) > 2: # incline trials with angle x incline interaction
                if refLimb == 'rH1':
                    mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) * speed + np.asarray(beta1[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1[nonpred]).reshape(-1,1) * param + np.asarray(beta1["pred2:pred3"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) + np.asarray(beta1[f"pred4{refLimb}"]).reshape(-1,1) #the mean of the third predictor (if present) is zero because it was centred before modelling
                    mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) * speed + np.asarray(beta2[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta2[nonpred]).reshape(-1,1) * param + np.asarray(beta2["pred2:pred3"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) + np.asarray(beta2[f"pred4{refLimb}"]).reshape(-1,1) 
                else:
                    mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) * speed + np.asarray(beta1[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1[nonpred]).reshape(-1,1) * param + np.asarray(beta1["pred2:pred3"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) #the mean of the third predictor (if present) is zero because it was centred before modelling
                    mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) * speed + np.asarray(beta2[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1[nonpred]).reshape(-1,1) * param + np.asarray(beta2["pred2:pred3"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) 
                phase2_preds[:,:] = np.arctan2(mu2, mu1)
                
            else: # head height trials with speed x angle interaction (param = speed, different percentiles)
                if refLimb == 'rH1':
                    mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) * param + np.asarray(beta1[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1["pred1:pred2"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) + np.asarray(beta1[f"pred3{refLimb}"]).reshape(-1,1)#the mean of the third predictor (if present) is zero because it was centred before modelling
                    mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) * param + np.asarray(beta2[pred]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T) + np.asarray(beta2["pred1:pred2"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) + np.asarray(beta2[f"pred3{refLimb}"]).reshape(-1,1)
                else:
                    mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) * param + np.asarray(beta1[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1["pred1:pred2"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) #the mean of the third predictor (if present) is zero because it was centred before modelling
                    mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) * param + np.asarray(beta2[pred]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T) + np.asarray(beta2["pred1:pred2"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param)
                phase2_preds[:,:] = np.arctan2(mu2, mu1)
            
            # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
            for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
                print(f"{refLimb}: working on data range {k}...")
                if k == 1:
                    phase2_preds[phase2_preds<0] = phase2_preds[phase2_preds<0]+2*np.pi
                if k == 2:
                    phase2_preds[phase2_preds<np.pi] = phase2_preds[phase2_preds<np.pi]+2*np.pi
                    if iprcnt == 2 and FigConfig.colour_config[clrs][iprcnt] not in legend_colours[i]:
                        legend_colours[i].append(FigConfig.colour_config[clrs][iprcnt])
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
                    ax[axid].fill_between(pred2_range + np.nanmean(pred2_relevant), 
                                          lower, 
                                          higher, 
                                          alpha = 0.2, 
                                          facecolor = FigConfig.colour_config[clrs][iprcnt]
                                          )
                    ax[axid].plot((pred2_range + np.nanmean(pred2_relevant)), 
                            trace, 
                            color = FigConfig.colour_config[clrs][iprcnt], 
                            linewidth = 1, 
                            linestyle = lnst, 
                            )
                
        #statistics 
        unique_traces_sub = unique_traces[np.where((unique_traces > ylim[0]) & (unique_traces < ylim[1]))[0]]
        unique_traces_sub_sorting = np.argsort(unique_traces_sub)
        unique_traces_sub_sorted = unique_traces_sub[unique_traces_sub_sorting]
        for i, (y_tr, y_tr_id) in enumerate(zip(unique_traces_sub_sorted, unique_traces_sub_sorting)):
            ax[axid].hlines(y = y_tr, 
                            xmin = np.nanmax(pred2_relevant) + 2, 
                            xmax = np.nanmax(pred2_relevant) + 4, 
                            linewidth = 0.5, 
                            color = 'black', 
                            linestyle = lnst)
            if i < len(unique_traces_sub)-1:
                ax[axid].vlines(x = np.nanmax(pred2_relevant) + 4, 
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
                
        ax[axid].text(np.mean(xlim)+((xlim[1]-xlim[0])*0.75/4)*(iprcnt-2), 1.25*np.pi, f"{prcnt}th", ha = 'center', color = FigConfig.colour_config[clrs][iprcnt])
             
    y_tr_spread = np.median(y_tr_list) + [-0.4,-0.2,0,0.2,0.4] # empirically determined that 0.2 is sufficient separation on my axis
    for isp in range(len(y_tr_spread)):
        ax[axid].text(x_tr_list[isp], 
                      y_tr_spread[isp],# + y_delta_list[isp], 
                      p_tr_list[isp])
    
    ax[axid].text(np.mean(xlim), 1.37*np.pi, pct_text, color = FigConfig.colour_config[clrs][0], ha = 'center')
    
    # axes 
    ax[axid].set_ylim(ylim[0], ylim[1])
    ax[axid].set_yticks(yticks)
    ax[axid].set_yticklabels(yticklabels)
    ax[axid].set_title(tlt, pad = 0)
    
ax[0].set_ylabel('Homolateral phase (rad)')

lgd = fig.legend([(legend_colours[0],legend_linestyles[0]), 
                  (legend_colours[1],legend_linestyles[1])], 
                ['left hindlimb', 'right hindlimb'],
                handler_map={tuple: AnyObjectHandlerDouble()}, loc = 'lower left',
                bbox_to_anchor=(0.25,0.95,0.5,0.3), mode="expand", borderaxespad=0.1,
                title = "Reference limb", ncol = 2)

plt.tight_layout(w_pad = 3)

figtitle = f"{yyyymmdd}_homolateral_COMBINED3.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight')
