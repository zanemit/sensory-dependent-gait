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

yyyymmdd = '2022-08-18'
outputDir = Config.paths['passiveOpto_output_folder']
datafrac = 0.3
iterations = 1000
ref = 'COMBINED'
limb = 'homolateral0'

appdx_dict = {3: '', 4: '_incline'}
sample_nums = {'_incline': 11713, '': 13206}

ylim = (0.3*np.pi,1.5*np.pi)
yticks = [0.5*np.pi,np.pi,1.5*np.pi]
yticklabels = ["0.5π", "π", "1.5π"]  

predictors = ['speed', 'snoutBodyAngle', 'incline', 'weight']
interaction = 'TRUEsecondary'
clrs = 'homolateral'
tlt = 'Incline trials'

# legend_colours = []
# legend_linestyles = []

fig, ax = plt.subplots(1,2,figsize = (1.65*2,1.8), sharey = True) #1.6,1.4 for 4figs S2 bottom row
for axid, param_col in enumerate([ 'snoutBodyAngle', 'incline'] ):
    
    sBA_split_str = 'FALSE'
        
    appdx = appdx_dict[len(predictors)]
    samples = sample_nums[appdx]
          
    if len(predictors) > 3:
        beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_interaction{interaction}_continuous_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_interaction{interaction}_continuous_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        # statspath = Path(outputDir) / f"{yyyymmdd}_coefCircCategorical_homolateral0_refCOMBINED_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"

    else: #head height trials
        beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        # statspath = Path(outputDir) / f"{yyyymmdd}_coefCircCategorical_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        
    beta1 = pd.read_csv(beta1path)
    beta2 = pd.read_csv(beta2path)
    # stats = pd.read_csv(statspath)

    datafull = data_loader.load_processed_data(dataToLoad = 'strideParams', 
                                               yyyymmdd = yyyymmdd,
                                               limb = ref, 
                                               appdx = appdx)[0]
    
    if 'deg' in datafull['headLVL'][0]:
        datafull['incline'] = [-int(x[3:]) for x in datafull['headLVL']]
        
    if len(predictors)>3 and param_col == 'snoutBodyAngle':
        predictor = 'incline'
        pred = 'pred3'
        nonpred = 'pred2'
        xlim = (-41,40)
        ax[axid].set_xlim(xlim[0], xlim[1])
        ax[axid].set_xticks([-40,-20,0,20,40])
        ax[axid].set_xlabel('Incline (deg)')
    else:
        predictor = 'snoutBodyAngle'
        pred = 'pred2'
        nonpred = 'pred3'
        xlim = (140,180)
        ax[axid].set_xlim(xlim[0],xlim[1])
        ax[axid].set_xticks([140,150,160,170,180])
        ax[axid].set_xlabel('Snout-hump angle (deg)')
    
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
        param = np.percentile(param_relevant, 50) - np.nanmean(param_relevant)
        
        weight_relevant = utils_processing.remove_outliers(datafull['weight'])
        weight = np.percentile(weight_relevant, prcnt) - np.nanmean(weight_relevant)
        print(weight)
        
        # initialise arrays
        phase2_preds = np.empty((beta1.shape[0], pred2_range.shape[0]))
        phase2_preds[:] = np.nan
        
        unique_traces = np.empty((0))
        
        if len(predictors) > 3: # incline trials with angle x incline interaction
            mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) * speed + np.asarray(beta1[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1[nonpred]).reshape(-1,1) * param +  np.asarray(beta1["pred2:pred3"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) + np.asarray(beta1['pred4']).reshape(-1,1) * weight #the mean of the third predictor (if present) is zero because it was centred before modelling
            mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) * speed + np.asarray(beta2[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1[nonpred]).reshape(-1,1) * param +  np.asarray(beta2["pred2:pred3"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) + np.asarray(beta2['pred4']).reshape(-1,1) * weight
            phase2_preds[:,:] = np.arctan2(mu2, mu1)
            
        else: # head height trials with speed x angle interaction
            mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) * param + np.asarray(beta1[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1["pred1:pred2"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) + np.asarray(beta1['pred3']).reshape(-1,1) * weight #the mean of the third predictor (if present) is zero because it was centred before modelling
            mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) * param + np.asarray(beta2[pred]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T) + np.asarray(beta2["pred1:pred2"]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T *param) + np.asarray(beta2['pred3']).reshape(-1,1) * weight
            phase2_preds[:,:] = np.arctan2(mu2, mu1)
        
        # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
        for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
            print(f"Working on data range {k}...")
            if k == 1:
                phase2_preds[phase2_preds<0] = phase2_preds[phase2_preds<0]+2*np.pi
            if k == 2:
                phase2_preds[phase2_preds<np.pi] = phase2_preds[phase2_preds<np.pi]+2*np.pi
                
            trace = scipy.stats.circmean(phase2_preds[:,:],high = hi, low = lo, axis = 0)
            lower = np.zeros_like(trace); higher = np.zeros_like(trace)
            for x in range(lower.shape[0]):
                lower[x], higher[x] =  utils_math.hpd_circular(phase2_preds[:,x], 
                                                                mass_frac = 0.95, 
                                                                high = hi, 
                                                                low = lo) #% (np.pi*2)
            
            if round(trace[-1],6) not in unique_traces and not np.any(abs(np.diff(trace))>5):
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
                        linestyle = 'solid', 
                        )
                                
        ax[axid].text(np.mean(xlim)+((xlim[1]-xlim[0])*0.75/4)*(iprcnt-2), 1.25*np.pi, f"{prcnt}th", ha = 'center', color = FigConfig.colour_config[clrs][iprcnt])
             
    ax[axid].text(np.mean(xlim), 1.37*np.pi, "weight percentile", color = FigConfig.colour_config[clrs][0], ha = 'center')
    
    # axes 
    ax[axid].set_ylim(ylim[0], ylim[1])
    ax[axid].set_yticks(yticks)
    ax[axid].set_yticklabels(yticklabels)
    ax[axid].set_title(tlt, pad = 0)
    
ax[0].set_ylabel('Homolateral phase (rad)')

plt.tight_layout(w_pad = 3)

figtitle = f"{yyyymmdd}_homolateral_COMBINED_weight_incline.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), dpi = 300)
