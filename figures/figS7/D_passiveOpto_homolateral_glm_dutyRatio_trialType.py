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

predictorlist = ['duty_ratio', 'snoutBodyAngle', 'incline']
predictorlist_str = ['duty_ratio', 'snout-hump angle', 'slope']
predictor = 'duty_ratio'
predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
appdx =  '_incline_COMBINEDtrialType'
tlt = 'All trials'
yyyymmdd = '2022-08-18'
slopes = ['pred2', 'pred3']
limb = 'lF0'
ref = 'lH1'
interaction = 'TRUEthreeway'
samples =  10395
datafrac = 0.3
iters = 1000
sbaSPLITstr = 'sBAsplitFALSE'

### PLOTTING
ylim = (0.3*np.pi,1.5*np.pi)
yticks = [0.5*np.pi,np.pi,1.5*np.pi]
yticklabels = ["0.5π", "π", "1.5π"]  
xlim, xticks, xlabel = treadmill_circGLM.get_predictor_range(predictor)
# xlim = (0.1,1)
# xticks = (0,0.5,1)

fig, ax = plt.subplots(1,1,figsize = (1.45,1.35)) #1.6,1.4 for 4figs S2 bottom row

last_vals = [] # for stats

for ref_id, (lnst, clr, lbl, xr) in enumerate(zip(['dashed', 'solid'],
                                            ['greys', 'homolateral'],
                                            ['head h.', 'slope'],
                                            [(143,173), (152,167)])):
    
    x_range, phase_preds = treadmill_circGLM.get_circGLM_slopes(
            predictors = predictorlist,
            yyyymmdd = yyyymmdd,
            limb = limb,
            ref = ref,
            categ_var = 'trialType',
            samples = samples,
            interaction = interaction,
            appdx = appdx,
            datafrac = datafrac,
            slopes = slopes,
            outputDir = Config.paths['passiveOpto_output_folder'],
            iterations = iters,
            x_pred_range = {"snoutBodyAngle": np.linspace(xr[0],xr[1],100)},
            mice = Config.passiveOpto_config['mice'],
            sBA_split_str =sbaSPLITstr,
            merged=True
                    ) 
    
    c = FigConfig.colour_config[clr][1+ref_id]
    pp = phase_preds[:, :, predictor_id, 0, ref_id]
    
    # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
    unique_traces = np.empty((0))
    for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
        print(f"Working on data range {k}...")
        if k == 1:
            pp[pp<0] = pp[pp<0]+2*np.pi
        if k == 2:
            pp[pp<np.pi] = pp[pp<np.pi]+2*np.pi
            ax.hlines(ylim[1]-0.3, 0.33+0.84*ref_id, 0.83+0.84*ref_id, color = c, ls = lnst, lw = 1)
            ax.text(xlim[0] + (0.01 * (xlim[1]-xlim[0])) + 0.84*ref_id,
                    ylim[1] - (0.05* (ylim[1]-ylim[0])),
                    lbl,
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
            ax.fill_between(x_range[:,predictor_id], 
                                  lower, 
                                  higher, 
                                  alpha = 0.25, 
                                  facecolor = c
                                  )
            ax.plot(x_range[:,predictor_id], 
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
        categ_var = 'trialType',
        samples = samples,
        interaction = interaction,
        appdx = appdx,
        datafrac = datafrac,
        slopes = slopes,
        outputDir = Config.paths['passiveOpto_output_folder'],
        iterations = iters,
        mice = Config.passiveOpto_config['mice'],
        sBA_split_str =sbaSPLITstr
                ) 

cat_coef_str = f"pred{len(predictorlist)+1}slope"
cont_coef_str = f"pred{predictor_id+1}"
# ax.text(x_range[-1, 1] + ((xlim[1]-xlim[0])/100),
#         np.mean(last_vals),
#         stat_dict[cat_coef_str])

ax.text(xlim[0] + (0.05 * (xlim[1]-xlim[0])),
        ylim[1] - (0.20* (ylim[1]-ylim[0])),
        f"Duty factor ratio: {stat_dict[cont_coef_str]}",
        # color=c,
        fontsize=5)

ax.text(xlim[0] + (0.36 * (xlim[1]-xlim[0])),
        ylim[1] - (0.05* (ylim[1]-ylim[0])),
        f"vs         trials: {stat_dict[cat_coef_str]}",
        fontsize=5)

# -------------------------------STATS-----------------------------------

ax.set_title(tlt)
    
# axes 
ax.set_xlim(xlim[0], xlim[1])
ax.set_xticks(xticks)
ax.set_xlabel("Hind-fore DF ratio")

ax.set_ylim(ylim[0], ylim[1])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_ylabel('Relative LF phase\n(rad)')

# -------------------------------LEGEND----------------------------------- 
# fig.legend(loc = 'center right', bbox_to_anchor=(1,0.65), fontsize=5)
# -------------------------------LEGEND----------------------------------- 
   
plt.tight_layout()

figtitle = f"passiveOpto_combined_trialType_duty_ratio.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
    

    
