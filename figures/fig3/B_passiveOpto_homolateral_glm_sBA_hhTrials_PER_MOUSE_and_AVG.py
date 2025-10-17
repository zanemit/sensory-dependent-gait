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

#---------------HEAD HEIGHT TRIALS--------------------
predictorlist = ['speed', 'snoutBodyAngle']
predictorlist_str = ['speed', 'snout-hump angle']
predictor = 'snoutBodyAngle'
predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
appdx = ''
samplenum = 9279
tlt = 'Head height trials'
yyyymmdd = '2022-08-18'
slopes = ['pred2']
limb = 'lF0'
datafrac = 1
ref = 'lH1combblncd'
interaction = 'TRUE'
mouselist = Config.passiveOpto_config['mice']
rfl_str = False

#---------------HEAD HEIGHT TRIALS--------------------


unique_traces = np.empty((0))

### PLOTTING
ylim = (0.3*np.pi,1.5*np.pi)
yticks = [0.5*np.pi,np.pi,1.5*np.pi]
yticklabels = ["0.5π", "π", "1.5π"]  
xlim, xticks, xlabel = treadmill_circGLM.get_predictor_range(predictor)
xlabel = 'Snout-hump angle\n(deg)'

#---------------PER MOUSE--------------------
fig, ax = plt.subplots(1,1,figsize = (1.35,1.35)) #1.6,1.4 for 4figs S2 bottom row
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
            outputDir = Config.paths['passiveOpto_output_folder'],
            iterations = 1000,
            mice = Config.passiveOpto_config['mice']
                    ) 
    
    c = np.random.choice(palette, 1)[0]
    pp = phase_preds[:, :, predictor_id, i, 0]
    xr = x_range[:, predictor_id, i, 0]
    print(i, xr[-1]-xr[0])
    # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
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
                    alpha = 0.8
                    )
#---------------PER MOUSE--------------------
  
#---------------AVERAGE--------------------            
xr_avg, phase_preds_avg = treadmill_circGLM.get_circGLM_slopes(
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
        outputDir = Config.paths['passiveOpto_output_folder'],
        iterations = 1000,
        mice = Config.passiveOpto_config['mice']
                ) 

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
        
        ax.plot(xr_avg[:,predictor_id], 
                trace_avg, 
                color = main_clr,
                linewidth = 1.5, 
                alpha = 1
                )
#---------------AVERAGE--------------------     

ax.set_title(tlt)
    
# axes 
ax.set_xlim(xlim[0], xlim[1])
ax.set_xticks(xticks[::2])
ax.set_xlabel(f"{xlabel}")

ax.set_ylim(ylim[0], ylim[1])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_ylabel('Left homolateral phase\n(rad)')

# -------------------------------STATS-----------------------------------
samplenum = 12023
limb = 'lF0'
datafrac = 0.7
ref = 'lH1comb'
categ_var='rH0_categorical'
interaction = 'TRUE'
stat_dict = treadmill_circGLM.get_circGLM_stats(
        predictors = predictorlist,
        yyyymmdd = yyyymmdd,
        limb = limb,
        ref = ref,
        categ_var = categ_var,
        samples = samplenum,
        interaction = interaction,
        appdx = appdx,
        datafrac = datafrac,
        slopes = slopes,
        outputDir = Config.paths['passiveOpto_output_folder'],
        iterations = 1000,
        mice = Config.passiveOpto_config['mice']
                ) 

cont_coef_str = f"pred{predictor_id+1}"

ax.text(xlim[0] + (0.1 * (xlim[1]-xlim[0])),
        ylim[1] - (0.05* (ylim[1]-ylim[0])),
        f"{predictorlist_str[predictor_id]}: {stat_dict[cont_coef_str]}",
        # color=c,
        fontsize=5)
# -------------------------------STATS-----------------------------------
    
plt.tight_layout()

figtitle = f"passiveOpto_{yyyymmdd}_{appdx}_{limb}_ref{ref}_{'_'.join(predictorlist)}_SLOPE{''.join(slopes)}_{interaction}_{appdx}_PER_MOUSE.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300,  
            bbox_inches = 'tight',
            transparent = True)
    

    
