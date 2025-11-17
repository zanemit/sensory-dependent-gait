import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.stats
import os

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

# PER-MOUSE, BUT A RESULT OF THE RANDOM SLOPE MODEL, NOT BETA12

import scipy.stats
from processing import data_loader, utils_math, utils_processing, treadmill_circGLM
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

# ------------- INCLINE TRIALS: NO SPEED, DUTY_RATIO -----------------
predictorlist = ['dutyratio', 'snoutBodyAngle', 'incline', 'strideLength']
predictorlist_str = ['duty_ratio', 'snout-hump angle', 'incline', 'stride length']
predictor = 'incline'#'duty_ref'#'strideLength' #'snoutBodyAngle'
predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
appdx =  '_incline'
tlt = 'Slope trials'
yyyymmdd = '2022-08-18'
slopes = ['pred2','pred3']
limb = 'homolateral0'
ref = 'COMBINEDcb'
ref_simple = 'COMBINED'
interaction = 'TRUE4way'
samples =  9083#12498
datafrac = 0.8#0.4
iters = 1000
sbaSPLITstr = 's'
# ------------- INCLINE TRIALS: NO SPEED, DUTY_RATIO -----------------


unique_traces = np.empty((0))

### LOAD FULL DATASET TO COMPUTE SPEED PERCENTILES
datafull = data_loader.load_processed_data(dataToLoad = 'strideParamsMerged',
                                           outputDir = Config.paths['passiveOpto_output_folder'],
                                           yyyymmdd = yyyymmdd,
                                           limb = ref_simple, 
                                           appdx = appdx)[0]

# sba = [145,160,175] #sba
# sba = [160]
# sba = [20,50,80] #speed
# sba=[0.5,0.75,1] #duty_ref
sba=[2,4,6] #stridelength
# sba=[0.5,1.1,1.7] #duty_ratio
prcnts = []
special_other_var = 'strideLength'#'duty_ratio'#'strideLength'#'duty_ref'#'snoutBodyAngle'
no_outliers_sba = utils_processing.remove_outliers(datafull[special_other_var])
for sb in sba:
    prcnts.append(scipy.stats.percentileofscore(no_outliers_sba, sb))

### PLOTTING
ylim = (0.3*np.pi,1.5*np.pi)
yticks = [0.5*np.pi,np.pi,1.5*np.pi]
yticklabels = ["0.5π", "π", "1.5π"]  
xlim, xticks, xlabel = treadmill_circGLM.get_predictor_range(predictor)

fig, ax = plt.subplots(1,1,figsize = (1.45,1.35)) #1.6,1.4 for 4figs S2 bottom row

last_vals = [] # for stats

# plot each mouse (just default ref limb)
for iprcnt, (prcnt, speed, lnst) in enumerate(zip(prcnts,
                                                  sba,
                                                  ['dotted', 'solid', 'dashed'])):

    c = FigConfig.colour_config['homolateral'][2*iprcnt]
    
    # get data for different speed percentiles
    x_range, phase_preds = treadmill_circGLM.get_circGLM_slopes(
            predictors = predictorlist,
            yyyymmdd = yyyymmdd,
            limb = limb,
            # categ_var = 'trialType',
            ref = ref,
            samples = samples,
            interaction = interaction,
            appdx = appdx,
            datafrac = datafrac,
            slopes = slopes,
            outputDir = Config.paths['passiveOpto_output_folder'],
            iterations = iters,
            mice = Config.passiveOpto_config['mice'],
            special_other_predictors = {special_other_var: prcnt},
            sBA_split_str =sbaSPLITstr,
            merged=True
                    ) 
   
    pp = phase_preds[:, :, predictor_id, 0, 0]
    # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
    for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
        print(f"Working on data range {k}...")
        if k == 1:
            pp[pp<0] = pp[pp<0]+2*np.pi
        if k == 2:
            pp[pp<np.pi] = pp[pp<np.pi]+2*np.pi
            ax.hlines(ylim[1]-0.7, -26.5+17*iprcnt, -15.5+17*iprcnt, color = c, ls = lnst, lw = 1)
            ax.text(xlim[0] + (0.15 * (xlim[1]-xlim[0])) + 17*iprcnt,
                    ylim[1] - (0.15* (ylim[1]-ylim[0])),
                    speed,
                    color=c,
                    fontsize=5)
            
        trace = scipy.stats.circmean(pp[:,:],high = hi, low = lo, axis = 0)
        lower = np.zeros_like(trace); higher = np.zeros_like(trace)
        for x in range(lower.shape[0]):
            lower[x], higher[x] =  utils_math.hpd_circular(pp[:,x], 
                                                            mass_frac = 0.95, 
                                                            high = hi, 
                                                            low = lo) #% (np.pi*2)
        
        if round(trace[-1],6) not in unique_traces and not np.any(abs(np.diff(trace))>5):
            unique_traces = np.append(unique_traces, round(trace[-1],6))
            print('plotting...')    
            ax.fill_between(x_range[:, predictor_id], 
                                  lower, 
                                  higher, 
                                  alpha = 0.15, 
                                  facecolor = c
                                  )
            ax.plot(x_range[:, predictor_id], 
                    trace, 
                    color = c,
                    linewidth = 1,
                    linestyle = lnst,
                    alpha = 1,
                    # label = lbl
                    )
        
            print(prcnt, trace[0], trace[-1])
        
        # for stats
        if trace[-1] > ylim[0] and trace[-1] < ylim[1] and trace[-1] not in last_vals:
            last_vals.append(trace[-1])

# -------------------------------STATS-----------------------------------
predictorlist = ['dutyrat', 'snoutBodyAngle', 'incline', 'strideLen']
samplenum =12030
datafrac = 0.4
ref = 'COMBINEDc'
categ_var='hmlg0_refLimb'
sba_str = 's'
stat_dict = treadmill_circGLM.get_circGLM_stats(
        predictors = predictorlist,
        yyyymmdd = yyyymmdd,
        limb = limb,
        ref = ref,
        samples = samplenum,
        categ_var = categ_var,
        interaction = interaction,
        appdx = appdx,
        datafrac = datafrac,
        slopes = slopes,
        outputDir = Config.paths['passiveOpto_output_folder'],
        iterations = iters,
        mice = Config.passiveOpto_config['mice'],
        sBA_split_str =sbaSPLITstr
                ) 

cont_coef_str = f"pred{predictor_id+1}"
print(stat_dict[cont_coef_str])
ax.text(xlim[0] + (0.1 * (xlim[1]-xlim[0])),
        ylim[1] - (0.03* (ylim[1]-ylim[0])),
        f"stride length x slope: {stat_dict['pred3:pred4']}",
        fontsize=5)

ax.text(xlim[0] + (0.75 * (xlim[1]-xlim[0])),
        ylim[1] - (0.15* (ylim[1]-ylim[0])),
        "cm",
        color="black",
        fontsize=5)


# -------------------------------STATS-----------------------------------

ax.set_title(tlt)
    
# axes 
ax.set_xlim(xlim[0], xlim[1])
ax.set_xticks(xticks)
xlabel = 'Surface slope (deg)'
ax.set_xlabel(xlabel)

ax.set_ylim(ylim[0], ylim[1])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_ylabel('Homolateral phase\n(rad)')

# -------------------------------LEGEND----------------------------------- 
# fig.legend(loc = 'center right', bbox_to_anchor=(1,0.65), fontsize=5)
# -------------------------------LEGEND----------------------------------- 
   
plt.tight_layout()

figtitle = f"passiveOpto_{limb}_ref{ref}_{'_'.join(predictorlist)}_SLOPE{''.join(slopes)}_{interaction}_{appdx}_AVERAGE_speed_percentiles.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
    

    
