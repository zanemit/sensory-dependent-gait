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


main_clr = FigConfig.colour_config['homolateral'][0]
palette = get_palette_from_html(main_clr,
                                lightness_values=[0.6,0.65,0.7, 0.75,0.8])

limb = 'lF0'
ref = 'lH1'
predictorlist = ['speed', 'snoutBodyAngle']
predictor = 'snoutBodyAngle'
predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
appdx = '_egr3'
tlt = 'MSA-deficient'
yyyymmdd = '2023-08-14'
yyyymmdd_fp = '2023-11-06'
slopes = ['pred2']
interaction = 'TRUE'
sample_num = 11584
data_frac = 1
rfl_str = False
mouselist = Config.passiveOpto_config['egr3_mice']

# sba_min = 140
# sba_max = 180
# sba_index = np.linspace(sba_min,
#                    sba_max,
#                    (sba_max-sba_min)*2+1,
#                     endpoint=True)

set_CoMy_ant = 0.5
set_CoMy_post = -1
set_CoMy_range_index = np.linspace(set_CoMy_ant,
                          set_CoMy_post,
                          num = abs(int(((set_CoMy_ant-set_CoMy_post)*32)+1)), 
                          endpoint=True)

confints_across_mice = pd.DataFrame(np.empty((set_CoMy_range_index.shape[0], 
                                              len(mouselist))) *np.nan,
                                    index = set_CoMy_range_index)

datafull = data_loader.load_processed_data(dataToLoad = 'strideParamsMerged', 
                                            yyyymmdd = yyyymmdd, 
                                            limb = ref, 
                                            appdx = appdx)[0]

#TODO need to align the data against SBA

# CoMy vs snout-hump angle 
CoMy_pred_path = os.path.join(
    Config.paths["forceplate_output_folder"],f"{yyyymmdd_fp}_mixedEffectsModel_linear_COMy_{predictor}.csv"
    )
CoMy_pred = pd.read_csv(CoMy_pred_path)
CoMy_pred_intercept = CoMy_pred.iloc[0,1]
CoMy_pred_slope = CoMy_pred.iloc[1,1]
# load data that the model is based on for centering of sBA data
data_fp = pd.read_csv(os.path.join(
    Config.paths["forceplate_output_folder"], f"{yyyymmdd_fp}_meanParamDF_{predictor}.csv")
    )

#---------------PER MOUSE--------------------
for i in range(1, len(mouselist)+1):
    # iterate over mice

    # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
    
    datafull_m = datafull[datafull['mouseID'] == mouselist[i-1]]
    sba_no_outliers = utils_processing.remove_outliers(datafull_m[predictor])
    sba_min_m = np.nanmin(sba_no_outliers)
    sba_max_m = np.nanmax(sba_no_outliers)
    
    # compute the corresponding CoMY range
    get_CoMy_ant = CoMy_pred_intercept + (CoMy_pred_slope*(sba_min_m-np.nanmean(datafull[predictor]))) + np.nanmean(data_fp["CoMy_mean"])
    get_CoMy_post = CoMy_pred_intercept + (CoMy_pred_slope*(sba_max_m-np.nanmean(datafull[predictor]))) + np.nanmean(data_fp["CoMy_mean"])
    
    CoMy_id_ant = np.argmin(np.abs(get_CoMy_ant-set_CoMy_range_index))
    CoMy_id_post = np.argmin(np.abs(get_CoMy_post-set_CoMy_range_index))
    CoMy_target_range = np.linspace(
        set_CoMy_range_index[CoMy_id_ant],
        set_CoMy_range_index[CoMy_id_post],
        CoMy_id_post-CoMy_id_ant+1,
        endpoint = True)
    
    sba_target_range = np.linspace(
        sba_min_m,
        sba_max_m,
        CoMy_target_range.shape[0],
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
    pp = phase_preds[:, :, predictor_id, i, 0]
    
    lo = -np.pi
    hi = np.pi
     
    trace = scipy.stats.circmean(pp[:,:],high = hi, low = lo, axis = 0)
    lower = np.zeros_like(trace); higher = np.zeros_like(trace)
    for x in range(lower.shape[0]):
        lower[x], higher[x] =  utils_math.hpd_circular(pp[:,x], 
                                                        mass_frac = 0.95, 
                                                        high = hi, 
                                                        low = lo) #% (np.pi*2)
    
    confints_across_mice.loc[CoMy_target_range,i-1] = higher-lower

fig, ax = plt.subplots(1,1, figsize = (1.35,1.35))
for i in range(len(mouselist)):
    c = np.random.choice(palette, 1)[0]
    ax.plot(confints_across_mice.index*Config.forceplate_config["fore_hind_post_cm"]/2, 
            confints_across_mice.iloc[:,i],
            lw = 0.7,
            color = c
            )
#---------------PER MOUSE--------------------    
    
#---------------ACROSS MICE--------------------
not_na_num = (~confints_across_mice.isna()).sum(axis=1)
mousenum_70pc = int(0.7*len(mouselist))
mousenum_mask = not_na_num >=mousenum_70pc
# ax.plot(confints_across_mice.index[mousenum_mask], 
#         confints_across_mice.mean(axis=1)[mousenum_mask],
#         lw = 1.5,
#         color = main_clr
#         )
series = pd.Series(confints_across_mice.values.ravel('F'))
indices = list(confints_across_mice.index) * confints_across_mice.shape[1]

from scipy.optimize import curve_fit
from scipy.stats import t

def quadratic(x,A,B,C):
    return A*x**2 + B*x + C

df = pd.DataFrame({'comy': indices, 'confint': series}).dropna()
popt,pcov = curve_fit(quadratic, df['comy'].values, df['confint'].values, p0=(0.1,0.1,0.1))
A_fit, B_fit, C_fit = popt
print(f"Quadratic fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}, C = {C_fit:.3f}")
x_pred = confints_across_mice.index[mousenum_mask]
ax.plot(x_pred*Config.forceplate_config["fore_hind_post_cm"]/2, 
        quadratic(x_pred, *popt),
        lw = 1.5,
        color = main_clr
        )
std_err = np.sqrt(np.diag(pcov)) # standard errors
t_values = popt/std_err
dof = max(0, df.shape[0]-len(popt))   
p_values = [2 * (1 - t.cdf(np.abs(t_val), dof)) for t_val in t_values]
print(f"p-values: A_p = {p_values[0]:.3e}, B_p = {p_values[1]:.3e}, k_p = {p_values[2]:.3e}")

ax.set_xlim(0.8,-1)#*limblen+3)
ax.set_ylim(0,0.3*np.pi)
# ax.set_yticks(np.linspace(0,1,6, endpoint=True))
ax.set_yticks([0,0.1*np.pi, 0.2*np.pi,0.3*np.pi])
ax.set_yticklabels(["0", "0.1π", "0.2π", "0.3π"])
ax.set_xticks([0.5,0,-0.5,-1], labels = ["0.5", "0", "-0.5", "-1"])
ax.set_ylabel("Confidence interval\n(rad)")
ax.set_title(tlt)
ax.set_xlabel("AP centre of\nsupport (cm)")
   
plt.tight_layout()

figtitle = f"passiveOpto_{limb}_confidence_int_vs_CoMY{appdx}.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)