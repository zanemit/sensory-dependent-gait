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


#-----------testing if there is a difference between per-mouse v slope model----
#-----------or if the discrepancy is due to the use of lH1 or COMB as reflimbs--
outcome_variable = 'limbSupportPC3' # should change
ref = 'lH1'
catvar = 'trialType'

yyyymmdd = '2022-05-06'
mouselist = set(Config.mtTreadmill_config['mice_level'] + Config.mtTreadmill_config['mice_incline'])
# ph_str = "shift"
appdx = "_COMBINEDtrialType"

if 'PC3' in outcome_variable:
    ytlt = "Diagonal support\n(PC3)"
elif "PC2" in outcome_variable:
    ytlt = "L-R sync or single-leg\nsupport (PC2)"
elif "PC4" in outcome_variable:
    ytlt = "Single-leg support\n(PC4)"
else:
    ytlt = "DEFINE LABEL!!!"

set_sba_ant = 135
set_sba_post = 180
set_sba_range_index = np.linspace(set_sba_ant,
                          set_sba_post,
                          num = abs(int((set_sba_post-set_sba_ant)+1)), 
                          endpoint=True)

support_preds_across_mice_dict = {}
slice_dict = {}

predictorlist = ['speed', 'snoutBodyAngle', 'incline']
slopes = ['pred2', 'pred3']
predictor = 'snoutBodyAngle'
predictor_str = 'snout-hump angle'
interaction = 'TRUEthreeway'

 
    
support_preds_across_mice = pd.DataFrame(np.empty((set_sba_range_index.shape[0], 
                                                 len(mouselist)))*np.nan,
                                       index = set_sba_range_index)

datafull = data_loader.load_processed_data(outputDir = Config.paths['mtTreadmill_output_folder'],
                                            dataToLoad = 'strideParamsMerged', 
                                            yyyymmdd = yyyymmdd, 
                                            limb = ref, 
                                            appdx = appdx)[0]
predictor_trdm = predictor
predictor_id = np.where(np.array(predictorlist)==predictor)[0][0]

#-------------PLOT!!-------------------  

if '3' in outcome_variable:
    yticks = np.linspace(0.2,1,5)
    ylims=(0.2, 1)
    fwidth = 1.5
elif '2' in outcome_variable:
    yticks = [-0.1, 0, 0.1] 
    ylims=(0.2, 1)
    fwidth = 1.5
elif '4' in outcome_variable:
    yticks = [-0.1, -0.05, 0, 0.05, 0.1] 
    ylims=(-0.1, 0.1)
    fwidth = 1.6
else:
    raise ValueError("ylim not specified!")
                              
fig, ax = plt.subplots(1,1, figsize = (fwidth,1.35)) 

    
for reflimb_id, (clr, lnst, lbl) in enumerate(zip(
        [FigConfig.colour_config['homolateral'][2], FigConfig.colour_config['greys'][1]],
        ['solid', 'dashed'],
        ['level', 'slope']
        )):
    # iterate over mice
    for im, m in enumerate(mouselist):
        # subset dataframe
        datafull_sub = datafull[datafull['mouseID'] == m]
        
        # set min/max predictor range that will be used for all mice
        values = utils_processing.remove_outliers(datafull_sub[predictor])
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
                outputDir = Config.paths['mtTreadmill_output_folder'],
                mice = mouselist,
                merged=True,
                x_pred_range = {predictor_trdm: x_target_range}# - np.nanmean(datafull_sub[predictor_trdm])}
                        ) 
        
        support_preds_across_mice.loc[x_target_range, im] = support_preds[:, predictor_id, im+1, reflimb_id]
        
    comy_pvals = []
    
       
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
    
    # add texts
    ax.text(set_sba_ant+2+(20*reflimb_id), 
            ylims[1]-((ylims[1]-ylims[0])*0.15),
            lbl,
            fontsize=5,
            color=clr)
    ax.hlines(ylims[1]-((ylims[1]-ylims[0])*0.155),
              set_sba_ant+2+(20*reflimb_id),
              set_sba_ant+12+(20*reflimb_id),
            ls=lnst,
            lw=0.7,
            color=clr)

#-----------STATS: IS PC3-vs-COS SLOPE DIFFERENT FROM ZERO?-------
ax.text(set_sba_ant+15, 
        ylims[1]-((ylims[1]-ylims[0])*0.15),
        "vs           trials:",
        fontsize=5,
        color='black')

mod_predictors = "_".join(predictorlist)
slopes_str = "".join(slopes)
mod_path = os.path.join(Config.paths['mtTreadmill_output_folder'], 
   f"{yyyymmdd}_contCoefficients_mixedEffectsModel_linear_{outcome_variable}_vs_{mod_predictors}_trialType_randSlopes{slopes_str}_interaction{interaction}.csv")

if os.path.exists(mod_path):
    stats = pd.read_csv(mod_path, index_col=0)
    p = stats.loc[f"pred{predictor_id+1}_centred", "Pr(>|t|)"]
    ptext = ""
    if (p < np.asarray(FigConfig.p_thresholds)).sum() == 0:
        ptext += "n.s."   
    else:
        ptext += '*' * (p < np.asarray(FigConfig.p_thresholds)).sum()
    
    ax.text(set_sba_ant + (0.1*(set_sba_post-set_sba_ant)),
            ylims[1] - (0.05* (ylims[1]-ylims[0])),
            f"{predictor_str}: {ptext}",
            fontsize=5)

# load fourway interaction data to get trial type - sba slope comparisons
# this analysis excluded some of the other interaction terms due to rank deficiency
interaction = 'TRUEfourway'
mod_path = os.path.join(Config.paths['mtTreadmill_output_folder'], 
   f"{yyyymmdd}_contCoefficients_mixedEffectsModel_linear_{outcome_variable}_vs_{mod_predictors}_trialType_randSlopes{slopes_str}_interaction{interaction}.csv")
if os.path.exists(mod_path):
    stats = pd.read_csv(mod_path, index_col=0)
    p_catvar = stats.loc[f"pred{predictor_id+1}_centred:{catvar}slope", "Pr(>|t|)"]
    ptext_catvar = '*' * (p_catvar < np.asarray(FigConfig.p_thresholds)).sum() if (p_catvar < np.asarray(FigConfig.p_thresholds)).sum()>0 else 'n.s.'
    ax.text(set_sba_ant + 45,
            ylims[1] - ((ylims[1]-ylims[0])*0.15),
            ptext_catvar,
            fontsize=5)
    
#-------------SAVE DICTS OR ADD STATS-------------------  

ax.set_xlim(set_sba_ant, set_sba_post) 
ax.set_xticks([140,160,180]) 
    
ax.set_ylim(ylims[0], ylims[1])
ax.set_yticks(yticks)
ax.set_ylabel(ytlt)   
ax.set_xlabel("Snout-hump angle\n(deg)†")   
 
plt.tight_layout()
    
figtitle = f"mtTreadmill_limbSupportPC{outcome_variable[-1]}_vs_CoS_incline_headheight.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
