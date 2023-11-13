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

outputDir = Config.paths['mtTreadmill_output_folder']
datafrac = 0.15
iterations = 1000
ref = 'COMBINED'
limb = 'homolateral0'
appdx = ''
interaction= 'FALSE'
pct_text = 'speed (cm/s)'
param_col = 'speed'

sample_nums = {2: 11428, 3: 13137}

ylim = (0.3*np.pi,1.5*np.pi)
yticks = [0.5*np.pi,np.pi,1.5*np.pi]
yticklabels = ["0.5π", "π", "1.5π"]  

# prcnts = [20,50,80]
prcnts = [24.02962306729136, 46.0780376038612, 77.93328930850215]

clrs = 'homolateral'

fig, ax = plt.subplots(1,1,figsize = (1.65,1.5)) #1.6,1.4 for 4figs S2 bottom row
yyyymmdd = '2022-05-06'
predictors = ['speed', 'snoutBodyAngle', 'incline']
nonparam_col = 'snoutBodyAngle'
tlt = 'Slope trials'

    
sBA_split_str = 'FALSE'
    
samples = sample_nums[len(predictors)]
      
if len(predictors) > 2:
    beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
    beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplit{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
    statspath2 = Path(outputDir) / f"{yyyymmdd}_coefCircContinuous_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
   
else: #head height trials
    beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
    beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
    statspath2 = Path(outputDir) / f"{yyyymmdd}_coefCircContinuous_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
    
beta1 = pd.read_csv(beta1path)
beta2 = pd.read_csv(beta2path)
stats2 = pd.read_csv(statspath2)

datafull = data_loader.load_processed_data(outputDir = outputDir,
                                           dataToLoad = 'strideParams', 
                                           yyyymmdd = yyyymmdd,
                                           limb = ref)[0]

if 'deg' in datafull['trialType'][0]:
    datafull['incline'] = [-int(x[3:]) for x in datafull['trialType']]
    
# if len(predictors)>2 and axid ==2 : # incline
predictor = 'incline'
pred = 'pred3' #incline
nonpred = 'pred2'
xlim = (-41,45)
ax.set_xlim(xlim[0], xlim[1])
ax.set_xticks([-40,-20,0,20,40])
ax.set_xlabel('Incline (deg)')
# else:
# predictor = 'snoutBodyAngle'
# pred = 'pred2'
# nonpred = 'pred3' #not used in head height trials
# xlim = (140,180)
# ax.set_xlim(xlim[0],xlim[1])
# ax.set_xticks([140,150,160,170,180])
# ax.set_xlabel('Snout-hump angle (deg)')

y_tr_list = np.empty((0)) # 5 percentiles to plot = 5 pairs to compare!
x_tr_list = np.empty((0))
p_tr_list = np.empty((0))
y_delta_list = np.empty((0))

speed_relevant = utils_processing.remove_outliers(datafull['speed'])
# speed = np.percentile(speed_relevant, 50) - np.nanmean(speed_relevant)
speed = prcnt - np.nanmean(speed_relevant)
    
# define the range of snout-hump angles or inclines (x axis predictor)
pred2_relevant = utils_processing.remove_outliers(datafull[predictor]) 
pred2_centred = pred2_relevant - np.nanmean(pred2_relevant) #centering
pred2_range = np.linspace(pred2_centred.min(), pred2_centred.max(), num = 100)
    
for iprcnt, prcnt in enumerate(prcnts):
    # find median param (excluding outliers)    
    param_relevant = utils_processing.remove_outliers(datafull[nonparam_col])
    nonparam = np.percentile(param_relevant, 50) - np.nanmean(param_relevant)
    
    speed_relevant = utils_processing.remove_outliers(datafull['speed'])
    speed = np.percentile(speed_relevant, prcnt) - np.nanmean(speed_relevant)
    
    # initialise arrays
    phase2_preds = np.empty((beta1.shape[0], pred2_range.shape[0]))
    phase2_preds[:] = np.nan
    
    unique_traces = np.empty((0))
    
    if len(predictors) > 2: # incline trials with angle x incline interaction
        mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) * speed + np.asarray(beta1[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1[nonpred]).reshape(-1,1) * nonparam 
        mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) * speed + np.asarray(beta2[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T + np.asarray(beta1[nonpred]).reshape(-1,1) * nonparam  
        phase2_preds[:,:] = np.arctan2(mu2, mu1)
        
    else: # head height trials with speed x angle interaction (param = speed, different percentiles)
        mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) + np.asarray(beta1['pred1']).reshape(-1,1) * speed + np.asarray(beta1[pred]).reshape(-1,1) @ pred2_range.reshape(-1,1).T  
        mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) + np.asarray(beta2['pred1']).reshape(-1,1) * speed + np.asarray(beta2[pred]).reshape(-1,1) @ (pred2_range.reshape(-1,1).T) 
        phase2_preds[:,:] = np.arctan2(mu2, mu1)
    
    # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
    for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
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
            unique_traces = np.append(unique_traces, round(trace[-1],6))
            print('plotting...')    
            ax.fill_between(pred2_range + np.nanmean(pred2_relevant), 
                                  lower, 
                                  higher, 
                                  alpha = 0.2, 
                                  facecolor = FigConfig.colour_config[clrs][iprcnt*2]
                                  )
            ax.plot((pred2_range + np.nanmean(pred2_relevant)), 
                    trace, 
                    color = FigConfig.colour_config[clrs][iprcnt*2], 
                    linewidth = 1
                    )
                  
    ax.text(np.mean(xlim)+((xlim[1]-xlim[0])*0.75/3)*(iprcnt-1), 1.25*np.pi, f"{prcnt:.0f}", ha = 'center', color = FigConfig.colour_config[clrs][iprcnt*2])
         
y_tr_spread = y_tr_list #np.median(y_tr_list) + [-0.4,-0.2,0,0.2,0.4] # empirically determined that 0.2 is sufficient separation on my axis
for isp in range(len(y_tr_spread)):
    ax.text(x_tr_list[isp], 
                  y_tr_spread[isp],# + y_delta_list[isp], 
                  p_tr_list[isp])

# SPEED STATS
param_of_interest_id = np.where(np.asarray(predictors) == param_col)[0][0]+1
row_for_param = np.where(stats2['Unnamed: 0'] == f'pred{param_of_interest_id} SSDO')[0][0]
signif = np.all((np.sign(stats2.loc[row_for_param, 'LB HPD']),np.sign(stats2.loc[row_for_param, 'UB HPD'])))
p_text = pct_text + ' ' + ('*' * signif)
if not signif:
    p_text += "n.s."  
    
ax.text(np.mean(xlim), 1.37*np.pi, p_text, color = FigConfig.colour_config[clrs][0], ha = 'center')


# axes 
ax.set_ylim(ylim[0], ylim[1])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_title(tlt, pad = 0)
    
ax.set_ylabel('Homolateral phase (rad)')

plt.tight_layout(w_pad = 3)

figtitle = f"MS2_{yyyymmdd}_homolateral_{predictor}.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            transparent = True
            # bbox_extra_artists = (lgd, ), 
            # bbox_inches = 'tight'
            )
