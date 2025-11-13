import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os 
from matplotlib import pyplot as plt
import scipy.stats

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

import scipy.stats
from processing import data_loader, utils_math, utils_processing, treadmill_linearGLM
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

outcome_variable = 'limbSupportPC1' # should change
ref = 'lH1'

yyyymmdd = '2022-08-18'
rH_category ="sync" #"sync" #"alt"

appdx = f"_rH{rH_category}"
catvar = None

yticks = np.linspace(-0.2,0.2,5)
ylims=(-0.2, 0.2)
ytexts = ('3-limb', 'hindlimb')

set_sba_ant = 135
set_sba_post = 180
set_sba_range_index = np.linspace(set_sba_ant,
                          set_sba_post,
                          num = abs(int((set_sba_post-set_sba_ant)+1)), 
                          endpoint=True)
predictor = 'snoutBodyAngle'
predictor_str = 'snout-hump angle'

predictor = 'snoutBodyAngle'
predictor_str = 'snout-hump angle'

category_dict = {'alt': 'alternation', 'sync': 'synchrony', 'asym': 'asymmetry'}

support_preds_across_mice_dict = {}
slice_dict = {}

predictorlist = ['speed', 'snoutBodyAngle']
slopes = ['pred1', 'pred2']
interaction = 'TRUE'

datafull = data_loader.load_processed_data(outputDir = Config.paths['passiveOpto_output_folder'],
                                            dataToLoad = 'strideParamsMerged', 
                                            yyyymmdd = yyyymmdd, 
                                            limb = ref, 
                                            appdx = appdx)[0]

predictor_trdm = predictor
predictor_id = np.where(np.array(predictorlist)==predictor)[0][0]
mouselist = datafull['mouseID'].unique()

support_preds_across_mice = pd.DataFrame(np.empty((set_sba_range_index.shape[0], 
                                                 len(mouselist)))*np.nan,
                                       index = set_sba_range_index)

#-------------PLOT!!------------------- 
fig, ax = plt.subplots(1,1, figsize = (1.25,1.35)) 

reflimb_id=0
clr= FigConfig.colour_config['homolateral'][2]
lnst='solid'
lbl='slope'

# iterate over mice
for im, m in enumerate(mouselist):
    # subset dataframe
    datafull_sub = datafull[datafull['mouseID'] == m]
    
    # set min/max predictor range that will be used for all mice
    if predictor=='snoutBodyAngle':
        values = utils_processing.remove_outliers(datafull_sub[predictor])
    else:
        values = datafull_sub[predictor]
    pred_min = np.nanmin(values)
    pred_max = np.nanmax(values)
    
    # find which ids of the set_pred_range these are the closest to!
    slice_dict[im] = (
        np.argmin(abs(set_sba_range_index-pred_min)),
        np.argmin(abs(set_sba_range_index-pred_max))
        )
        
    x_target_range = np.linspace(
        set_sba_range_index[slice_dict[im][0]],
        set_sba_range_index[slice_dict[im][1]],
        abs(slice_dict[im][1]-slice_dict[im][0])+1,
        endpoint = True)
    
    x_preds, support_preds = treadmill_linearGLM.get_linear_slopes(
            predictors = predictorlist,
            outcome_variable = outcome_variable,
            yyyymmdd = yyyymmdd,
            ref = ref,
            interaction = interaction,
            appdx = appdx,
            categ_var=catvar,
            slopes = slopes,
            outputDir = Config.paths['passiveOpto_output_folder'],
            mice = mouselist,
            merged=True,
            x_pred_range = {predictor_trdm: x_target_range}# - np.nanmean(datafull_sub[predictor_trdm])}
                    ) 
    
    support_preds_across_mice.loc[x_target_range, im] = support_preds[:, predictor_id, im+1, reflimb_id]
    
ax.set_title(f"{category_dict[rH_category]}", fontsize=6)
   
#-------------COMPUTE SHIFTS---------------------------

mask = support_preds_across_mice.isna().sum(axis=1)<(support_preds_across_mice.shape[1]/2)
med = np.mean(support_preds_across_mice.loc[mask,:], axis=1)
std = np.std(support_preds_across_mice.loc[mask,:], axis=1)

ax.fill_between(set_sba_range_index[mask],
                med+std,#-med_first_nonan, 
                med-std,#-med_first_nonan,
                alpha = 0.2, 
                facecolor = clr)
ax.plot(set_sba_range_index[mask],
        med,#-med_first_nonan, 
        color = clr,
        ls = lnst,
        lw = 1.5)

mod_predictors = "_".join(predictorlist)
slopes_str = "".join(slopes)
mod_path = os.path.join(Config.paths['passiveOpto_output_folder'], 
   f"{yyyymmdd}_contCoefficients_mixedEffectsModel_linear_rH{rH_category}_{outcome_variable}_vs_{mod_predictors}_randSlopes{slopes_str}_interaction{interaction}.csv")

if os.path.exists(mod_path):
    stats = pd.read_csv(mod_path, index_col = 0)
    p1 = stats.loc[f'pred{np.argwhere(np.asarray(predictorlist)==predictor)[0][0]+1}_centred', "Pr(>|t|)"]
    for i, p in enumerate([p1]):
        ptext = ""
        if (p < np.asarray(FigConfig.p_thresholds)).sum() == 0:
            ptext += "n.s."   
        else:
            ptext += '*' * (p < np.asarray(FigConfig.p_thresholds)).sum()
    
        ax.text(set_sba_ant + (0.5*(set_sba_post-set_sba_ant)),
                ylims[1] - (0.05* (ylims[1]-ylims[0])),
                f"{predictor_str.split(' ')[-1]}: {ptext}", ha='center',
                fontsize=5)

speed_effect = stats.loc[f"pred{np.argwhere(np.asarray(predictorlist)=='speed')[0][0]+1}_centred", "Estimate"]
pred_effect = stats.loc[f"pred{np.argwhere(np.asarray(predictorlist)==predictor)[0][0]+1}_centred", "Estimate"]
print(f"Speed effect per deg:\n{speed_effect}")
print(f"Predictor effect per deg:\n{pred_effect}")
print(f"Speed effect as % of pred effect: {abs(speed_effect)*100/abs(pred_effect):.1f}%")

# add y labels
ax.text(set_sba_ant-(set_sba_post-set_sba_ant)*0.1,
        ylims[0]-(ylims[1]-ylims[0])*0.15,
        ytexts[0],
        fontsize=5,
        ha='right', fontweight='bold',
        color=FigConfig.colour_config['homolateral'][1])
ax.text(set_sba_ant-(set_sba_post-set_sba_ant)*0.1,
        ylims[1]+(ylims[1]-ylims[0])*0.1,
        ytexts[1],
        fontsize=5,
        ha='right',fontweight='bold',
        color=FigConfig.colour_config['homologous'][1])
    
#-------------SAVE DICTS OR ADD STATS-------------------  

if predictor=='snoutBodyAngle':
    ax.set_xlim(set_sba_ant, set_sba_post) 
    ax.set_xticks([140,160,180]) 
elif predictor=='incline':
    ax.set_xlim(set_sba_ant, set_sba_post) 
    ax.set_xticks([-40,0,40]) 
    
ax.set_ylim(ylims[0], ylims[1])
ax.set_yticks(yticks)
ax.set_ylabel(outcome_variable[-3:])   
ax.set_xlabel(f"{predictor_str.capitalize()}\n(deg)")   
 
plt.tight_layout()
    
figtitle = f"passiveOpto_{rH_category}_{outcome_variable}_vs_sba.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
