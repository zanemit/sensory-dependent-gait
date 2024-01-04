import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from processing import data_loader, utils_processing
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

fig, ax = plt.subplots(1,1, figsize = (0.9,1.15))

limb = 'lF0'#,'rF0','rH0']
refLimb = 'lH1'
iters = 1000
yyyymmdd = '2022-08-18'

u = 1 
lw = 1.5  

tsize = 6

# CoMy vs snoutBodyAngle regression (none of the mice of interest, so use the average)
CoMy_snoutBodyAngle_path = os.path.join(Config.paths["forceplate_output_folder"],"2021-10-26_mixedEffectsModel_linear_COMy_snoutBodyAngle.csv")
CoMy_snoutBodyAngle = pd.read_csv(CoMy_snoutBodyAngle_path)
CoMy_snoutBodyAngle_intercept = CoMy_snoutBodyAngle.iloc[0,1]
CoMy_snoutBodyAngle_slope = CoMy_snoutBodyAngle.iloc[1,1]
# load data that the model is based on for centering of sBA data
datafp1 = pd.read_csv(os.path.join(Config.paths["forceplate_output_folder"], "2021-10-26_meanParamDF_snoutBodyAngle.csv"))
    
# CoMy vs incline regression 2022-04-04 (not all of the same mice)
CoMy_incline_path = os.path.join(Config.paths["forceplate_output_folder"], "2022-04-04_mixedEffectsModel_linear_COMy_levels.csv")
CoMy_incline = pd.read_csv(CoMy_incline_path)
CoMy_incline_intercept = CoMy_incline.iloc[0,1]
CoMy_incline_slope = CoMy_incline.iloc[1,1]
# load data that the model is based on for centering of sBA data
datafp2 = pd.read_csv(os.path.join(Config.paths["forceplate_output_folder"], "2022-04-04_meanParamDF_levels.csv"))
      
# phase ~ speed + snoutBodyAngle + incline
glm_inclineTrials_path = os.path.join(Config.paths["passiveOpto_output_folder"], f"{yyyymmdd}_beta12acrossMice_{limb}_ref{refLimb}_speed_snoutBodyAngle_incline_interactionTRUEsecondary_{iters}its_100burn_3lag.csv")
glm_inclineTrials = pd.read_csv(glm_inclineTrials_path) 
glm_inclineTrials = glm_inclineTrials.iloc[:,1:]
mice_inclineTrials = np.unique([x[:10] for x in glm_inclineTrials.columns])   
# load data that the model is based on
datafull1 = data_loader.load_processed_data(dataToLoad = 'strideParams', 
                                            yyyymmdd = yyyymmdd, 
                                            limb = refLimb, 
                                            appdx = '_incline')[0]
datafull1['incline'] = [-int(x[3:]) for x in datafull1['headLVL']]

# phase ~ speed + snoutBodyAngle
glm_headHeightTrials_path = os.path.join(Config.paths["passiveOpto_output_folder"], f"{yyyymmdd}_beta12acrossMice_{limb}_ref{refLimb}_speed_snoutBodyAngle_interactionTRUE_{iters}its_100burn_3lag.csv")
glm_headHeightTrials = pd.read_csv(glm_headHeightTrials_path) 
mice_headHeightTrials = np.unique([x[:10] for x in glm_headHeightTrials.columns])
# load data that the model is based on
datafull2 = data_loader.load_processed_data(dataToLoad = 'strideParams', 
                                            yyyymmdd = yyyymmdd, 
                                            limb = refLimb, 
                                            appdx = "")[0]

mice = np.intersect1d(mice_inclineTrials, mice_headHeightTrials)

phase1_incT_sBA = []
phase2_incT_sBA = []
phase1_incT_inc = []
phase2_incT_inc = []
phase1_hhT_sBA = []
phase2_hhT_sBA = []
# bad mice FAA1034949 and FAA1034942! (no incline effect)

pred_num = 100
phase_preds_across_mice = np.empty((mice.shape[0], pred_num, 3))

for im, m in enumerate(mice):
    # subset dataframe
    datafull_sub1 = datafull1[datafull1['mouseID'] == m]
    datafull_sub2 = datafull2[datafull2['mouseID'] == m]
    
    # compute the median speed
    speed1_relevant = utils_processing.remove_outliers(datafull_sub1['speed'])
    speed1 = np.nanmedian(speed1_relevant) - np.nanmean(speed1_relevant)
    
    speed2_relevant = utils_processing.remove_outliers(datafull_sub2['speed'])
    speed2 = np.nanmedian(speed2_relevant) - np.nanmean(speed2_relevant)
    
    incline_relevant = utils_processing.remove_outliers(datafull_sub1['incline'])
    incline = np.nanmedian(incline_relevant) - np.nanmean(incline_relevant)
    
    angle1_relevant = utils_processing.remove_outliers(datafull_sub1['snoutBodyAngle'])
    angle1 = np.nanmedian(angle1_relevant) - np.nanmean(angle1_relevant)
    
    angle2_relevant = utils_processing.remove_outliers(datafull_sub2['snoutBodyAngle'])
    angle2 = np.nanmedian(angle2_relevant) - np.nanmean(angle2_relevant)
    
    # compute the snoutBodyAngle range in incline trials
    sBA_min = np.nanmin(datafull_sub1['snoutBodyAngle']) #- np.nanmean(datafull1['snoutBodyAngle'])
    sBA_max = np.nanmax(datafull_sub1['snoutBodyAngle']) #- np.nanmean(datafull1['snoutBodyAngle'])
    sBA_range = np.linspace(sBA_min, sBA_max, pred_num, endpoint = True)
    
    # compute the corresponding CoMy change (adding means to de-center CoMy) -> CoMy is as in the Fig1 plot
    CoMy_sBAmin = (CoMy_snoutBodyAngle_intercept + (sBA_min - np.nanmean(datafp1['param'])) * CoMy_snoutBodyAngle_slope) + np.nanmean(datafp1['CoMy_mean'])
    CoMy_sBAmax = (CoMy_snoutBodyAngle_intercept + (sBA_max - np.nanmean(datafp1['param'])) * CoMy_snoutBodyAngle_slope) + np.nanmean(datafp1['CoMy_mean'])
    
    # find the corresponding inclines (add the dataset mean to de-center the values)
    incline_sBAmin = ((CoMy_sBAmin - np.nanmean(datafp2['CoMy_mean']) - CoMy_incline_intercept) / CoMy_incline_slope) + np.nanmean(datafp2['param'])
    incline_sBAmax = ((CoMy_sBAmax - np.nanmean(datafp2['CoMy_mean']) - CoMy_incline_intercept) / CoMy_incline_slope) + np.nanmean(datafp2['param'])
    incline_range = np.linspace(incline_sBAmin, incline_sBAmax, pred_num, endpoint= True)
    
    for i, (phaselist1, phaselist2, title) in enumerate(zip([phase1_incT_sBA,phase1_incT_inc, phase1_hhT_sBA], 
                                                            [phase2_incT_sBA, phase2_incT_inc, phase2_hhT_sBA], 
                                                            ["incline trials: body tilt", "incline trials: incline", "head height trials: body tilt"]
                                                            )):
        if i == 0: #body angle incline trials
            mu1 = np.asarray(glm_inclineTrials[f'{m}_beta1_intercept']).reshape(-1,1) + \
                    np.asarray(glm_inclineTrials[f'{m}_beta1_speed']).reshape(-1,1) * speed1 + \
                    np.asarray(glm_inclineTrials[f'{m}_beta1_snoutBodyAngle']).reshape(-1,1) * (sBA_range - np.nanmean(datafull_sub1['snoutBodyAngle'])).reshape(-1,1).T + \
                    np.asarray(glm_inclineTrials[f'{m}_beta1_incline']).reshape(-1,1) * incline + \
                    np.asarray(glm_inclineTrials[f'{m}_beta1_snoutBodyAngle_incline']).reshape(-1,1) * incline * (sBA_range - np.nanmean(datafull_sub1['snoutBodyAngle'])).reshape(-1,1).T
            mu2 = np.asarray(glm_inclineTrials[f'{m}_beta2_intercept']).reshape(-1,1) + \
                    np.asarray(glm_inclineTrials[f'{m}_beta2_speed']).reshape(-1,1) * speed1 + \
                    np.asarray(glm_inclineTrials[f'{m}_beta2_snoutBodyAngle']).reshape(-1,1) * (sBA_range - np.nanmean(datafull_sub1['snoutBodyAngle'])).reshape(-1,1).T + \
                    np.asarray(glm_inclineTrials[f'{m}_beta2_incline']).reshape(-1,1) * incline + \
                    np.asarray(glm_inclineTrials[f'{m}_beta2_snoutBodyAngle_incline']).reshape(-1,1) * incline * (sBA_range - np.nanmean(datafull_sub1['snoutBodyAngle'])).reshape(-1,1).T
                
        elif i == 1: #incline incline trials
            mu1 = np.asarray(glm_inclineTrials[f'{m}_beta1_intercept']).reshape(-1,1) + \
                    np.asarray(glm_inclineTrials[f'{m}_beta1_speed']).reshape(-1,1) * speed1 + \
                    np.asarray(glm_inclineTrials[f'{m}_beta1_snoutBodyAngle']).reshape(-1,1) * angle1 + \
                    np.asarray(glm_inclineTrials[f'{m}_beta1_incline']).reshape(-1,1) * (incline_range - np.nanmean(datafull_sub1['incline'])).reshape(-1,1).T + \
                    np.asarray(glm_inclineTrials[f'{m}_beta1_snoutBodyAngle_incline']).reshape(-1,1) * angle1 * (incline_range - np.nanmean(datafull_sub1['incline'])).reshape(-1,1).T
                
            mu2 = np.asarray(glm_inclineTrials[f'{m}_beta2_intercept']).reshape(-1,1) + \
                    np.asarray(glm_inclineTrials[f'{m}_beta2_speed']).reshape(-1,1) * speed1 + \
                    np.asarray(glm_inclineTrials[f'{m}_beta2_snoutBodyAngle']).reshape(-1,1) * angle1 + \
                    np.asarray(glm_inclineTrials[f'{m}_beta2_incline']).reshape(-1,1) * (incline_range - np.nanmean(datafull_sub1['incline'])).reshape(-1,1).T   + \
                    np.asarray(glm_inclineTrials[f'{m}_beta2_snoutBodyAngle_incline']).reshape(-1,1) * angle1 * (incline_range - np.nanmean(datafull_sub1['incline'])).reshape(-1,1).T
        
        else: #body angle head height trials
            mu1 = np.asarray(glm_headHeightTrials[f'{m}_beta1_intercept']).reshape(-1,1) + \
                    np.asarray(glm_headHeightTrials[f'{m}_beta1_speed']).reshape(-1,1) * speed2 + \
                    np.asarray(glm_headHeightTrials[f'{m}_beta1_snoutBodyAngle']).reshape(-1,1) * (sBA_range - np.nanmean(datafull_sub2['snoutBodyAngle'])).reshape(-1,1).T + \
                    np.asarray(glm_headHeightTrials[f'{m}_beta1_speed_snoutBodyAngle']).reshape(-1,1) * (sBA_range - np.nanmean(datafull_sub2['snoutBodyAngle'])).reshape(-1,1).T * speed2
                    
            mu2 = np.asarray(glm_headHeightTrials[f'{m}_beta2_intercept']).reshape(-1,1) + \
                    np.asarray(glm_headHeightTrials[f'{m}_beta2_speed']).reshape(-1,1) * speed2 + \
                    np.asarray(glm_headHeightTrials[f'{m}_beta2_snoutBodyAngle']).reshape(-1,1) * (sBA_range - np.nanmean(datafull_sub2['snoutBodyAngle'])).reshape(-1,1).T   + \
                    np.asarray(glm_headHeightTrials[f'{m}_beta2_speed_snoutBodyAngle']).reshape(-1,1) * (sBA_range - np.nanmean(datafull_sub2['snoutBodyAngle'])).reshape(-1,1).T * speed2
                      
        phase_preds = np.arctan2(mu2, mu1)
        mean_phase = scipy.stats.circmean(phase_preds, high = np.pi, low = -np.pi, axis=0)
        if np.any(abs(np.diff(mean_phase))>5):
            phase_preds[phase_preds<0] = phase_preds[phase_preds<0]+2*np.pi
            mean_phase = scipy.stats.circmean(phase_preds, high = 2*np.pi, low = 0, axis=0)
        
        phase_preds_across_mice[im, :, i] = mean_phase
        # phaselist1.append(mean_phase[0])   
        # phaselist2.append(mean_phase[-1]) 

phase_preds_across_mice = phase_preds_across_mice - phase_preds_across_mice[:, 0, :].reshape(12,1,3)
phase_preds_sum = np.empty((mice.shape[0], pred_num, 2))
phase_preds_sum[:,:,1] =  phase_preds_across_mice[:,:,2]
phase_preds_sum[:,:,0] =  phase_preds_across_mice[:,:,0] + phase_preds_across_mice[:,:,1]

stds = np.nanstd(phase_preds_sum, axis = 0)

CoMy_range = np.linspace(CoMy_sBAmin, CoMy_sBAmax, pred_num)
meds = []
for i, (clr, lbl) in enumerate(zip([FigConfig.colour_config['homolateral'][2], FigConfig.colour_config['greys7'][0]],
                                 [ 'Surface slope', 'Head height'])): #iterate over conditions
    med = np.nanmedian(phase_preds_sum[:,:,i], axis=0)
    meds.append(med[-1])
    ax.fill_between(CoMy_range,
                    med+stds[:,i], 
                    med-stds[:,i],
                    alpha = 0.2, 
                    facecolor = clr)
    ax.plot(CoMy_range,
            med, 
            color = clr,
            lw = 1.5,
            label = lbl)
    
    ax.hlines(y = med[-1], 
                    xmin = CoMy_range[-1] -0.01, 
                    xmax = CoMy_range[-1] -0.02, 
                    linewidth = 0.5, 
                    color = 'black',
                    )
        
ax.set_xlim(-0.05, -0.52)    
ax.set_xticks(np.linspace(-0.1,-0.5,5)[[0,2,4]] )
ax.tick_params(axis = 'both', which = 'major', labelsize = tsize) 
ax.set_ylim(-np.pi,0.5*np.pi)
ax.set_yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi])
ax.set_yticklabels(["-π", "-0.5π", "0", "0.5π"], size = tsize)

ax.set_ylabel("Homolateral phase\nshift (rad)", size = tsize)
ax.set_xlabel("Anteroposterior CoS\nshift (cm)", size = tsize)

ax.vlines(x = CoMy_range[-1]-0.02, 
                ymin = meds[0], 
                ymax = meds[1], 
                linewidth = 0.5, 
                color = 'black'
                ) 
ax.text(CoMy_range[-1]-0.03, np.mean(meds)-0.1,"n.s.", ha = 'left', size = tsize) 
lgd = plt.legend(bbox_to_anchor=(0.05,0.8,1.2,0.3), 
            mode="expand", borderaxespad=0.1, 
            # title = 'Modulated parameter',
            fontsize =tsize,
            # title_fontsize = tsize
            )
# plt.tight_layout()
fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"MS3_{yyyymmdd}_mouseComparisonIncline_interaction2_{limb}_LINEPLOT_FOR_BRISTOL.svg", 
            dpi=300,
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight'
            )





