import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.stats
import os

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

# TODO: THIS CODE SHOULD BE SPLIT TO PLOT THE TWO SUBPLOTS SEPARATELY

import scipy.stats
from processing import data_loader, utils_math, utils_processing, treadmill_circGLM
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

predictorlist = ['speed', 'snoutBodyAngle', 'incline']
predictorlist_str = ['speed', 'snout-hump angle', 'incline']
appdx =  '_incline'
tlt = 'Slope trials'
yyyymmdd = '2022-08-18'
slopes = ['pred2', 'pred3']
limb = 'homolateral0'
ref = 'COMBINED'
interaction = 'TRUEthreeway'
samples = 15618#12373
datafrac = 0.4
iters = 1000

### LOAD FULL DATASET TO COMPUTE SPEED PERCENTILES
datafull = data_loader.load_processed_data(dataToLoad = 'strideParamsMerged',
                                           outputDir = Config.paths['passiveOpto_output_folder'],
                                           yyyymmdd = yyyymmdd,
                                           limb = ref, 
                                           appdx = appdx)[0]
datafull['incline'] = [-int(lvl[3:]) for lvl in datafull['headLVL']]

### PLOTTING
ylim = (0.3*np.pi,1.5*np.pi)
yticks = [0.5*np.pi,np.pi,1.5*np.pi]
yticklabels = ["0.5π", "π", "1.5π"] 
hline_text_dict = {'snoutBodyAngle': (145, 151, 9), 'incline': (-27.5, -16.5, 17)}

fig, ax = plt.subplots(1,2,figsize = (1.15*2,1.35), sharey=True) #1.6,1.4 for 4figs S2 bottom row 
# fig, ax = plt.subplots(1,2,figsize = (1.1*2,1.35), sharey=True) #1.6,1.4 for 4figs S2 bottom row 

for i, (sba, predictor, prcnt_predictor, xlbl) in enumerate(zip(
        [[-40,0,40], [145,160,175]],
        ['snoutBodyAngle', 'incline'],
        ['incline', 'snoutBodyAngle'],
        ['Snout-hump angle (deg)', 'Surface slope (deg)']
        )):
    unique_traces = np.empty((0))
    
    predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
    
    prcnts = []
    no_outliers_sba = utils_processing.remove_outliers(datafull[prcnt_predictor])
    for sb in sba:
        prcnts.append(scipy.stats.percentileofscore(no_outliers_sba, sb))
    
    xlim, xticks, xlabel = treadmill_circGLM.get_predictor_range(predictor)
    
    last_vals = [] # for stats
    
    
    # plot each mouse (just default ref limb)
    for iprcnt, (prcnt, speed, lnst) in enumerate(zip(prcnts,
                                                      sba,
                                                      ['dotted', 'solid', 'dashed'])):
        prcnt_predictor_dict = {prcnt_predictor: prcnt}
        c = FigConfig.colour_config['homolateral'][2*iprcnt]
        
        # get data for different speed percentiles
        x_range, phase_preds = treadmill_circGLM.get_circGLM_slopes(
                predictors = predictorlist,
                yyyymmdd = yyyymmdd,
                limb = limb,
                ref = ref,
                samples = samples,
                interaction = interaction,
                appdx = appdx,
                datafrac = datafrac,
                slopes = slopes,
                outputDir = Config.paths['passiveOpto_output_folder'],
                iterations = iters,
                mice = Config.passiveOpto_config['mice'],
                special_other_predictors = prcnt_predictor_dict
                        ) 
        pp = phase_preds[:, :, predictor_id, 0, 0]
    
        # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
        for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
            print(f"Working on data range {k}...")
            if k == 1:
                pp[pp<0] = pp[pp<0]+2*np.pi
            if k == 2:
                pp[pp<np.pi] = pp[pp<np.pi]+2*np.pi
                ax[i].hlines(ylim[1]-0.7, 
                             hline_text_dict[predictor][0] + hline_text_dict[predictor][2]*iprcnt, 
                             hline_text_dict[predictor][1] + hline_text_dict[predictor][2]*iprcnt, 
                             color = c, ls = lnst, lw = 1)
                ax[i].text(xlim[0] + (0.15 * (xlim[1]-xlim[0])) + hline_text_dict[predictor][2]*iprcnt,
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
                ax[i].fill_between(x_range[:, predictor_id], 
                                      lower, 
                                      higher, 
                                      alpha = 0.15, 
                                      facecolor = c
                                      )
                ax[i].plot(x_range[:, predictor_id], 
                        trace, 
                        color = c,
                        linewidth = 1,
                        linestyle = lnst,
                        alpha = 1,
                        # label = lbl
                        )
            
            # for stats
            if trace[-1] > ylim[0] and trace[-1] < ylim[1] and trace[-1] not in last_vals:
                last_vals.append(trace[-1])
    
    # -------------------------------STATS-----------------------------------
    stat_dict = treadmill_circGLM.get_circGLM_stats(
            predictors = predictorlist,
            yyyymmdd = yyyymmdd,
            limb = limb,
            ref = ref,
            samples = samples,
            interaction = interaction,
            appdx = appdx,
            datafrac = datafrac,
            slopes = slopes,
            outputDir = Config.paths['passiveOpto_output_folder'],
            iterations = iters,
            mice = Config.passiveOpto_config['mice']
                    ) 
    
    cont_coef_str = f"pred{predictor_id+1}"
    
    ax[i].text(xlim[0] + (0.22 * (xlim[1]-xlim[0])),
            ylim[1] - (0.03* (ylim[1]-ylim[0])),
            f"angle x slope: {stat_dict['pred2:pred3']}",
            fontsize=5)
    
    ax[i].text(xlim[0] + (0.75 * (xlim[1]-xlim[0])),
            ylim[1] - (0.15* (ylim[1]-ylim[0])),
            "deg",
            color="grey",
            fontsize=5)


    # -------------------------------STATS-----------------------------------

    ax[i].set_title(tlt)
    
    xticks = xticks[::2] if predictor == 'snoutBodyAngle' else xticks
    xlbl = 'Snout-hump angle\n(deg)' if predictor == 'snoutBodyAngle' else 'Surface slope\n(deg)'
    
    # axes 
    ax[i].set_xlim(xlim[0], xlim[1])
    ax[i].set_xticks(xticks)
    ax[i].set_xlabel(xlbl)

ax[0].set_ylim(ylim[0], ylim[1])
ax[0].set_yticks(yticks)
ax[0].set_yticklabels(yticklabels)
ax[0].set_ylabel('Homolateral phase\n(rad)')

# -------------------------------LEGEND----------------------------------- 
# fig.legend(loc = 'center right', bbox_to_anchor=(1,0.65), fontsize=5)
# -------------------------------LEGEND----------------------------------- 
   
plt.tight_layout()

figtitle = f"passiveOpto_{limb}_ref{ref}_{'_'.join(predictorlist)}_SLOPE{''.join(slopes)}_{interaction}_{appdx}_AVERAGE_speed_percentiles.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
    

    
