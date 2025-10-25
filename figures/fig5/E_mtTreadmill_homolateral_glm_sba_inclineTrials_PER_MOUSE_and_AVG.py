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
from figures.fig_config import AnyObjectHandler, get_palette_from_html

main_clr = FigConfig.colour_config['homolateral'][1]
palette = get_palette_from_html(main_clr,
                                lightness_values=[0.6,0.65,0.7,0.75,0.8])

#---------------INCLINE TRIALS--------------------
predictorlist = ['speed', 'snoutBodyAngle', 'incline']
predictorlist_str = ['speed', 'snout-hump angle', 'slope']
predictor = 'snoutBodyAngle'
predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
appdx =  ''
tlt = 'Slope trials'
yyyymmdd = '2022-05-06'
slopes = ['pred2', 'pred3']
limb = 'lF0'
ref = 'lH1LleadRleadaltblncd'
interaction = 'TRUEthreeway'
samplenum =  7317
datafrac = 1
iters = 1000
mouselist = Config.mtTreadmill_config['mice_incline']

#---------------INCLINE TRIALS--------------------

#---------------PER MOUSE--------------------
### PLOTTING
ylim = (0.3*np.pi,1.5*np.pi)
yticks = [0.5*np.pi,np.pi,1.5*np.pi]
yticklabels = ["0.5π", "π", "1.5π"]  
xlim, xticks, xlabel = treadmill_circGLM.get_predictor_range(predictor)

fig, ax = plt.subplots(1,1,figsize = (1.35,1.4)) #1.6,1.4 for 4figs S2 bottom row
for i in range(1, len(mouselist)+1):
    x_range, phase_preds = treadmill_circGLM.get_circGLM_slopes_per_mouse(
            predictors = predictorlist,
            yyyymmdd = yyyymmdd,
            limb = limb,
            ref = ref,
            samples =samplenum,
            interaction = interaction,
            appdx = appdx,
            datafrac = datafrac,
            categ_var = None,
            slopes = slopes,
            outputDir = Config.paths['mtTreadmill_output_folder'],
            iterations = 1000,
            mice = mouselist
                    ) 
    
    c = np.random.choice(palette, 1)[0]
    pp = phase_preds[:, :, predictor_id, i, 0]
    xr = x_range[:, predictor_id, i, 0]
    # print(i, xr[-1]-xr[0])
    # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
    unique_traces = np.empty((0))
    for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
        print(f"Working on data range {k}...")
        if k == 1:
            pp[pp<0] = pp[pp<0]+2*np.pi
        if k == 2:
            pp[pp<np.pi] = pp[pp<np.pi]+2*np.pi
            
        trace = scipy.stats.circmean(pp[:,:],high = hi, low = lo, axis = 0)
        
        if round(trace[-1],6) not in unique_traces and not np.any(abs(np.diff(trace))>5):
            unique_traces = np.append(unique_traces, round(trace[-1],6))
            print('plotting...')    
            
            ax.plot(xr, 
                    trace, 
                    color = c,
                    linewidth = 0.7, 
                    alpha = 0.8,
                    zorder=1
                    )

#---------------AVERAGE TRIALS-------------------- 
for ymd, predlist, samples, appendix, intr, dfrac, slp, clr, z, lnst, trialtype, multip in zip(
        ['2021-10-23', yyyymmdd],
        [["speed", "snoutBodyAngle"], predictorlist],
        [15366, samplenum],
        ["", appdx],
        ["TRUE", interaction],
        [1, datafrac],
        [["pred2"], slopes],
        [FigConfig.colour_config["greys"][1], main_clr],
        [0,2],
        ['dashed','solid'],
        ['level', 'slope'],
        [0.47, 0.03]
        ):
    x_range_avg, phase_preds_avg = treadmill_circGLM.get_circGLM_slopes(
            predictors = predlist,
            yyyymmdd = ymd,
            limb = limb,
            ref = ref,
            categ_var = None,
            samples = samples,
            interaction = intr,
            appdx = appendix,
            datafrac = dfrac,
            slopes = slp,
            outputDir = Config.paths['mtTreadmill_output_folder'],
            iterations = 1000,
            mice = mouselist
                    ) 
    predictor_id = np.where(np.asarray(predlist) == predictor)[0][0]        
    unique_traces = np.empty((0))           
    pp_avg = phase_preds_avg[:, :, predictor_id, 0, 0]            
    for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
        print(f"Working on data range {k}...")
        if k == 1:
            pp_avg[pp_avg<0] = pp_avg[pp_avg<0]+2*np.pi
        if k == 2:
            pp_avg[pp_avg<np.pi] = pp_avg[pp_avg<np.pi]+2*np.pi
            
        trace_avg = scipy.stats.circmean(pp_avg[:,:],high = hi, low = lo, axis = 0)
        
        if round(trace_avg[-1],6) not in unique_traces and not np.any(abs(np.diff(trace_avg))>5):
            print('plotting...')    
            
            ax.plot(x_range_avg[:, predictor_id], 
                    trace_avg, 
                    color = clr,
                    linewidth = 1.5, 
                    alpha = 1,
                    linestyle=lnst,
                    zorder = z
                    )
            
    ax.text(xlim[0] + (multip * (xlim[1]-xlim[0])),
            ylim[1] - (0.15* (ylim[1]-ylim[0])),
            trialtype,
            color= clr,
            fontsize=5)
    ax.hlines(ylim[1]-0.7, 159-19*z/2, 170-19*z/2, 
              color = clr, ls = lnst, lw = 1)
#---------------AVERAGE--------------------

ax.set_title(tlt)
    
# axes 
ax.set_xlim(xlim[0], xlim[1])
ax.set_xticks(xticks[::2])
ax.set_xlabel("Snout-hump angle\n(deg)†")

ax.set_ylim(ylim[0], ylim[1])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_ylabel('Left homolateral phase\n(rad)')

# -------------------------------STATS-----------------------------------
stat_dict = treadmill_circGLM.get_circGLM_stats(
        predictors = predictorlist,
        yyyymmdd = yyyymmdd,
        limb = limb,
        ref = ref,
        categ_var = None,
        samples = samplenum,
        interaction = interaction,
        appdx = appdx,
        datafrac = datafrac,
        slopes = slopes,
        outputDir = Config.paths['mtTreadmill_output_folder'],
        iterations = iters,
        mice = mouselist
                ) 

cont_coef_str = f"pred{predictor_id+1}"
# cat_coef_str = f"pred{len(predictorlist)+1}slope"

ax.text(xlim[0] + (0.15 * (xlim[1]-xlim[0])),
        ylim[1] - (0.05* (ylim[1]-ylim[0])),
        f"{predictorlist_str[predictor_id]}: {stat_dict[cont_coef_str]}",
        # color=c,
        fontsize=5)

ax.text(xlim[0] + (0.33 * (xlim[1]-xlim[0])),
        ylim[1] - (0.15* (ylim[1]-ylim[0])),
        f"vs          trials",
        # color=c,
        fontsize=5)

# -------------------------------STATS-----------------------------------
    
plt.tight_layout()

figtitle = f"mtTreadmill_{yyyymmdd}_{appdx}_{limb}_ref{ref}_{'_'.join(predictorlist)}_SLOPE{''.join(slopes)}_{interaction}_{appdx}_PER_MOUSE.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300,  
            bbox_inches = 'tight',
            transparent = True)
    

    
