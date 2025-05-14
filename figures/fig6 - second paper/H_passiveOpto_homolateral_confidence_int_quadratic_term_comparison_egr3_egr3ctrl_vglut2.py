import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.stats
from scipy.stats import spearmanr
import os

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

# PER-MOUSE, BUT A RESULT OF THE RANDOM SLOPE MODEL, NOT BETA12

import scipy.stats
from processing import data_loader, utils_math, utils_processing, treadmill_circGLM
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler, get_palette_from_html

# REMOVE THESE IMPORTS AND FUNC TO COMPUTE CONFINT AMPLITUDE
from scipy.optimize import curve_fit
from scipy.stats import t

def quadratic(x,A,B,C):
    return A*x**2 + B*x + C

main_clr = FigConfig.colour_config['homolateral'][0]
palette = get_palette_from_html(main_clr,
                                lightness_values=[0.6,0.65,0.7,0.75,0.8])


predictorlist = ['speed', 'snoutBodyAngle']
predictor = 'snoutBodyAngle'
predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
tlt = 'MSA-deficient'
slopes = ['pred2']
interaction = 'TRUE'
rfl_str = False

limb = 'lF0'
ref = 'lH1'
samplenums = [11584, 14667, 13110]
datafracs = [1, 0.6, 0.6]
reflimb_strs = [False, False, False]
appdxs = ['_egr3', '_egr3ctrl', '']

sba_min = 140
sba_max = 180
sba_index = np.linspace(sba_min,
                   sba_max,
                   (sba_max-sba_min)*2+1,
                    endpoint=True)

average_confints = {}
for i, (yyyymmdd, appdx, sample_num, data_frac, mouselist, rfl_str) in enumerate(zip(
            ['2023-08-14', '2023-09-21', '2022-08-18'],
            appdxs,
            samplenums,
            datafracs,
            [Config.passiveOpto_config['egr3_mice'], Config.passiveOpto_config['egr3_ctrl_mice'], Config.passiveOpto_config['mice']],
            reflimb_strs
            )):
    
    average_confints[appdx] = [] # REMOVE THIS  LINETO COMPUTE CONFINT AMPLITUDE
    confints_across_mice = pd.DataFrame(np.empty((sba_index.shape[0], 
                                                  len(mouselist))) *np.nan,
                                        index = sba_index)
    
    datafull = data_loader.load_processed_data(dataToLoad = 'strideParamsMerged', 
                                                yyyymmdd = yyyymmdd, 
                                                limb = ref, 
                                                appdx = appdx)[0]
    
    unique_traces = np.empty((0))
    
    #---------------PER MOUSE--------------------
    # for i in range(1, len(mouselist)+1):
    for i,m in enumerate(mouselist):
        # iterate over mice
    
        # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
        
        datafull_m = datafull[datafull['mouseID'] == m]
        sba_no_outliers = utils_processing.remove_outliers(datafull_m[predictor])
        sba_min_m = np.nanmin(sba_no_outliers)
        sba_max_m = np.nanmax(sba_no_outliers)
        sba_id_min = np.argmin(np.abs(sba_min_m-sba_index))
        sba_id_max = np.argmin(np.abs(sba_max_m-sba_index))
        
        sba_target_range = np.linspace(
            sba_index[sba_id_min],
            sba_index[sba_id_max],
            int((sba_index[sba_id_max]-sba_index[sba_id_min])*2)+1,
            endpoint = True)
        
        x_range, phase_preds = treadmill_circGLM.get_circGLM_slopes_per_mouse(
                predictors = predictorlist,
                yyyymmdd = yyyymmdd,
                limb = limb,
                ref = ref,
                samples = sample_num,
                interaction = interaction,
                appdx = appdx,
                categ_var=None,
                datafrac = data_frac,
                slopes = slopes,
                outputDir = Config.paths['passiveOpto_output_folder'],
                iterations = 1000,
                mice = mouselist,
                x_pred_range = {predictor: sba_target_range - np.nanmean(datafull_m[predictor])}
                        ) 
        pp = phase_preds[:, :, predictor_id, i+1, 0]
        
        # lo = -np.pi
        # hi = np.pi
         
        for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
            print(f"Working on data range {k}...")
            if k == 1:
                pp[pp<0] = pp[pp<0]+2*np.pi
            if k == 2:
                pp[pp<np.pi] = pp[pp<np.pi]+2*np.pi
                
            trace = scipy.stats.circmean(pp[:,:],high = hi, low = lo, axis = 0)
            
            if round(trace[-1],6) not in unique_traces and not np.any(abs(np.diff(trace))>5):
                unique_traces = np.append(unique_traces, round(trace[-1],6))
                
                lower = np.zeros_like(trace); higher = np.zeros_like(trace)
                for x in range(lower.shape[0]):
                    lower[x], higher[x] =  utils_math.hpd_circular(pp[:,x], 
                                                                    mass_frac = 0.95, 
                                                                    high = hi, 
                                                                    low = lo) #% (np.pi*2)
                
                confints_across_mice.loc[sba_target_range[0]:sba_target_range[-1],i] = higher-lower
    
    # REMOVE THE FOR LOOP TO COMPUTE CONFINT AMPLITUDE
    for im, m in enumerate(mouselist):
         df_quadratic_m = pd.DataFrame({'sba': confints_across_mice.index, 'confint': confints_across_mice.iloc[:,im]}).dropna()
         popt,pcov = curve_fit(quadratic, df_quadratic_m['sba'].values, df_quadratic_m['confint'].values, p0=(0.1,0.1,0.1))
         A_fit, B_fit, C_fit = popt           
         average_confints[appdx].append(A_fit)
                
    # average_confints[appdx] = np.nanmax(confints_across_mice, axis=0)-np.nanmin(confints_across_mice, axis=0)
 #%%          
#-----------------------PLOT------------------------------
fig, ax = plt.subplots(1,1, figsize = (1.05,1.3))
for i, (clr, key) in enumerate(zip(
        [FigConfig.colour_config['homolateral'][2], FigConfig.colour_config['greys'][1], FigConfig.colour_config['greys'][1]],
        appdxs
        )):
    ax.boxplot(
                # list(confints_unique),
                average_confints[key],
                positions = [i+1], 
                medianprops = dict(color = clr, linewidth = 1, alpha = 0.6),
                boxprops = dict(color = clr, linewidth = 1, alpha = 0.6), 
                capprops = dict(color = clr, linewidth = 1, alpha = 0.6),
                whiskerprops = dict(color = clr, linewidth = 1, alpha = 0.6), 
                flierprops = dict(mec = clr, linewidth = 1, alpha = 0.6, ms=2)
                )
    
    ax.scatter(np.repeat(i+1.2, len(average_confints[key])),
                # confints_unique,
                average_confints[key],
                color = clr,
                s=1
                 )

ax.set_xlim(0.5,3.5)#*limblen+3)
ax.set_ylim(-0.0005,0.004)
ax.set_yticks(np.linspace(0,0.004,5, endpoint=True))
ax.set_yticklabels(np.arange(5))
ax.set_xticks([1,2,3])#,4,5,7,8])
ax.set_xticklabels(["MSA","CTRL","vGlut2"], rotation=45)#,4,5,7,8])

### -------------------------STATS-------------------------------------
_, p1 = scipy.stats.ttest_ind(average_confints[appdxs[0]], average_confints[appdxs[1]])
print(f"Egr3 vs Egr3ctrl: p-val {p1}")
_, p2 = scipy.stats.ttest_ind(average_confints[appdxs[0]], average_confints[appdxs[2]])
print(f"Egr3 vs vGlut2: p-val {p2}")
_, p3 = scipy.stats.ttest_ind(average_confints[appdxs[1]], average_confints[appdxs[2]])
print(f"Egr3ctrl vs vGlut2: p-val {p3}")

ptext = []
for i, (p, pos, ydelta) in enumerate(zip([p1,p3,p2], 
                                         [1.5,2.6,2],
                                         [0,0.0,0.0005])):
    if (p < np.asarray(FigConfig.p_thresholds)/3).sum() == 0:
        ptext.append( "n.s." )    
    else:
        ptext.append('*' * (p < np.asarray(FigConfig.p_thresholds)/3).sum())
    ax.text(pos,
            0.00325+ydelta,
            ptext[i], 
            ha = 'center',
            fontsize=5)
    
ax.hlines(0.0032, 1.1, 1.9, color = 'black', linewidth = 0.25)
ax.hlines(0.0032, 2.1, 3, color = 'black', linewidth = 0.25)
ax.hlines(0.0036, 1.1, 3, color = 'black', linewidth = 0.25)
### -------------------------STATS-------------------------------------

ax.set_ylabel("Rate of change in CI\nrange " + r"($10^{-3}\ rad/deg^2$)")    
plt.tight_layout()

figtitle = f"passiveOpto_{limb}_confidence_int_diff_comparison_egr3_egr3ctrl_vglut2.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)