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

predictorlist = ['speed', 'snoutBodyAngle', 'incline']
predictorlist_str = ['speed', 'snout-hump angle', 'slope']
predictor = 'speed'
predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
appdx = '_incline'
samples = 7726#7790
tlt = 'Slope trials'
yyyymmdd = '2022-08-18' 
slopes = ['pred2', 'pred3']
limb = 'rH0'
datafrac = 1
ref = 'lH1altadvancedblncd'
ref_simple = 'lH1'
interaction = 'TRUEthreeway'
rfl_str = None
sba_str = 'sBAsplitFALSE_FLIPPED'
iters = 1000

unique_traces = np.empty((0))

mice_unilateral_inj = Config.injection_config['right_inj_imp'] + Config.injection_config['left_inj_imp'] 
mouselist = np.intersect1d(Config.passiveOpto_config['mice'], mice_unilateral_inj)

### LOAD FULL DATASET TO COMPUTE SPEED PERCENTILES
datafull = data_loader.load_processed_data(dataToLoad = 'strideParamsMerged',
                                           outputDir = Config.paths['passiveOpto_output_folder'],
                                           yyyymmdd = yyyymmdd,
                                           limb = ref_simple, 
                                           appdx = appdx)[0]

sbas = [153,159,165]
prcnts = []
no_outliers_speed = utils_processing.remove_outliers(datafull['snoutBodyAngle'])
for sp in sbas:
    prcnts.append(scipy.stats.percentileofscore(no_outliers_speed, sp))

### PLOTTING
ylim = (-0.5*np.pi,1.5*np.pi)
yticks = [-0.5*np.pi, 0, 0.5*np.pi,np.pi,1.5*np.pi]
yticklabels = ["-0.5π", "0", "0.5π", "π", "1.5π"]  
xlim, xticks, xlabel = treadmill_circGLM.get_predictor_range(predictor)
xticks = [0,50,100,150]
xlabel = ' '.join(xlabel.split(' ')[:-1]) + '\n' + xlabel.split(' ')[-1]

fig, ax = plt.subplots(1,1,figsize = (1.35,1.35)) #1.6,1.4 for 4figs S2 bottom row

last_vals = [] # for stats


# plot each mouse (just default ref limb)
for iprcnt, (prcnt, speed, lnst) in enumerate(zip(prcnts,
                                                  sbas,
                                                  ['dotted', 'solid', 'dashed'])):

    c = FigConfig.colour_config['homologous'][2*iprcnt]
    
    # get data for different speed percentiles
    x_range, phase_preds = treadmill_circGLM.get_circGLM_slopes(
            predictors = predictorlist,
            yyyymmdd = yyyymmdd,
            limb = limb,
            ref = ref,
            samples = samples,
            categ_var = rfl_str,
            interaction = interaction,
            appdx = appdx,
            datafrac = datafrac,
            slopes = slopes,
            outputDir = Config.paths['passiveOpto_output_folder'],
            iterations = iters,
            mice = mouselist,
            special_other_predictors = {'snoutBodyAngle': prcnt},
            sBA_split_str = sba_str
                    ) 
   
    x_range = x_range[4:]; phase_preds = phase_preds[:, 4:]
    
    pp = phase_preds[:, :, predictor_id, 0, 0]
    # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
    for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
        print(f"Working on data range {k}...")
        if k == 1:
            pp[pp<0] = pp[pp<0]+2*np.pi
        if k == 2:
            pp[pp<np.pi] = pp[pp<np.pi]+2*np.pi
            ax.hlines(ylim[1]-1.01, 40+35*iprcnt, 64+35*iprcnt, color = c, ls = lnst, lw = 1)
            ax.text(xlim[0] + (0.25 * (xlim[1]-xlim[0])) + 35*iprcnt,
                    ylim[1] - (0.13* (ylim[1]-ylim[0])),
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
                                  alpha = 0.1, 
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
            print(trace[0], trace[-1])
        
        # for stats
        if trace[-1] > ylim[0] and trace[-1] < ylim[1] and trace[-1] not in last_vals:
            last_vals.append(trace[-1])

# -------------------------------STATS-----------------------------------
samples = 10923
datafrac = 1 
ref = 'lH1altadvanced'
sba_str = 'sFLIPPED'
categ_var = 'lF0_categorical'
stat_dict = treadmill_circGLM.get_circGLM_stats(
        predictors = predictorlist,
        yyyymmdd = yyyymmdd,
        limb = limb,
        ref = ref,
        samples = samples,
        interaction = interaction,
        appdx = appdx,
        datafrac = datafrac,
        categ_var = categ_var,
        slopes = slopes,
        outputDir = Config.paths['passiveOpto_output_folder'],
        iterations = iters,
        mice = mouselist,
        sBA_split_str = sba_str
                ) 

# cont_coef_str = f"pred{predictor_id+1}"
# ax.text(x_range[-1, 1] + ((xlim[1]-xlim[0])/100),
#         np.mean(last_vals),
#         stat_dict[cat_coef_str])

ax.text(xlim[0] + (0.15 * (xlim[1]-xlim[0])),
        ylim[1] - (0.03* (ylim[1]-ylim[0])),
        f"{predictorlist_str[0]} x angle: {stat_dict['pred1:pred2']}",
        fontsize=5)
# ax.text(xlim[0] + (0.4 * (xlim[1]-xlim[0])),
#         ylim[1] - (0.13* (ylim[1]-ylim[0])),
#         f"{predictorlist_str[0]}: {stat_dict[cont_coef_str]}",
#         fontsize=5)

ax.text(xlim[0] + (0.93 * (xlim[1]-xlim[0])),
        ylim[1] - (0.13* (ylim[1]-ylim[0])),
        "deg",
        color="grey",
        fontsize=5)


# -------------------------------STATS-----------------------------------

ax.set_title(tlt)
    
# axes 
ax.set_xlim(xlim[0], xlim[1])
ax.set_xticks(xticks)
ax.set_xlabel(f"{xlabel}")

ax.set_ylim(ylim[0], ylim[1])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_ylabel('Hindlimb phase\n(rad)')

# -------------------------------LEGEND----------------------------------- 
# fig.legend(loc = 'center right', bbox_to_anchor=(1,0.65), fontsize=5)
# -------------------------------LEGEND----------------------------------- 
   
plt.tight_layout()

figtitle = f"passiveOpto_{limb}_ref{ref}_{'_'.join(predictorlist)}_SLOPE{''.join(slopes)}_{interaction}_{appdx}_AVERAGE_speed_percentiles.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
    

    
