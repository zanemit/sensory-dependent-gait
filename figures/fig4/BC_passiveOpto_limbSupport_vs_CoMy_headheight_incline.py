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
outcome_variable = 'limbSupportPC4' # should change
ref = 'lH1'
catvar = None

yyyymmdd = '2022-08-18'
mouselist = Config.passiveOpto_config['mice']
# ph_str = "shift"
appdx = "_incline_COMBINEDtrialType"

if 'PC3' in outcome_variable:
    ytlt = "diagonal\nsupport (PC3)"
elif "PC2" in outcome_variable:
    ytlt = "L-R sync or single-leg\nsupport (PC2)"
elif "PC4" in outcome_variable:
    ytlt = "single-leg\nsupport (PC4)"
else:
    ytlt = "DEFINE LABEL!!!"

set_CoMy_ant = 0.5
set_CoMy_post = -1
set_CoMy_range_index = np.linspace(set_CoMy_ant,
                          set_CoMy_post,
                          num = abs(int(((set_CoMy_ant-set_CoMy_post)*64)+1)), 
                          endpoint=True)
set_CoMy_range_index_str = set_CoMy_range_index*Config.forceplate_config["fore_hind_post_cm"]/2

support_preds_across_mice_dict = {}
slice_dict = {}
for i, (yyyymmdd_fp, predictorlist, slopes, predictor, interaction, dict_lbl) in enumerate(zip(
           ['2021-10-26', '2022-04-04', '2021-10-26'],
           [['speed', 'snoutBodyAngle', 'incline'], ['speed', 'snoutBodyAngle', 'incline'], ['speed', 'snoutBodyAngle']],
           [['pred2', 'pred3'], ['pred2', 'pred3'], ['pred2']],
           ['snoutBodyAngle', 'levels', 'snoutBodyAngle'],
           ['TRUEthreeway', 'TRUEthreeway', 'TRUE'],
           ['incline_sBA', 'incline_incline', 'hh_sBA']
        )):  
    
    support_preds_across_mice = pd.DataFrame(np.empty((set_CoMy_range_index.shape[0], 
                                                     len(mouselist)))*np.nan,
                                           index = set_CoMy_range_index)
    
    datafull = data_loader.load_processed_data(dataToLoad = 'strideParamsMerged', 
                                                yyyymmdd = yyyymmdd, 
                                                limb = ref, 
                                                appdx = appdx)[0]
    
    if predictor == 'levels':
        predictor_id = np.where(np.asarray(predictorlist) == "incline")[0][0]
        datafull['incline'] = [-int(x[3:]) for x in datafull['headLVL']]
        predictor_trdm = 'incline'
        datafull = datafull[datafull['trialType']=='slope']
    else:
        predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
        predictor_trdm = predictor
        if 'incline' in predictorlist:
            datafull = datafull[datafull['trialType']=='slope']
        else:
            datafull = datafull[datafull['trialType']=='headHeight']
             
    # CoMy vs snout-hump angle
    if yyyymmdd_fp=='2022-04-04':
        CoMy_pred_path = os.path.join(
            Config.paths["forceplate_output_folder"],f"{yyyymmdd_fp}_mixedEffectsModel_linear_COMy_{predictor}.csv"
            )
    else:
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
    
    # compute the corresponding sBA range 
    # i.e. what angles correspond to the CoMy range in the dataframe
    get_pred_ant = ((set_CoMy_ant-np.nanmean(data_fp['CoMy_mean']) - CoMy_pred_intercept) / CoMy_pred_slope) + np.nanmean(data_fp['param']) 
    get_pred_post = ((set_CoMy_post-np.nanmean(data_fp['CoMy_mean']) - CoMy_pred_intercept) / CoMy_pred_slope) + np.nanmean(data_fp['param']) 
    
    set_pred_range = np.linspace(get_pred_ant, 
                                 get_pred_post, 
                                 support_preds_across_mice.shape[0], 
                                 endpoint= True)
    print(get_pred_ant, get_pred_post)
    # iterate over mice
    for im, m in enumerate(mouselist):
        # subset dataframe
        datafull_sub = datafull[datafull['mouseID'] == m]
        
        # set min/max predictor range that will be used for all mice
        if i == 0: # for 2022 dataset, this should be the snoutBodyAngle from incline trials
            pred_min = np.nanmin(datafull_sub[predictor])
            pred_max = np.nanmax(datafull_sub[predictor])
            
            # find which ids of the set_pred_range these are the closest to!
            slice_dict[im] = (
                np.argmin(abs(set_pred_range-pred_min)),
                np.argmin(abs(set_pred_range-pred_max))
                )
            
        x_target_range = np.linspace(
            set_pred_range[slice_dict[im][0]],
            set_pred_range[slice_dict[im][1]],
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
                x_pred_range = {predictor_trdm: x_target_range}# - np.nanmean(datafull_sub[predictor_trdm])}
                        ) 
        
        support_preds_sub = support_preds[:, predictor_id, im+1, 0]
        
        
        CoMy_indices = np.linspace(
            set_CoMy_range_index[slice_dict[im][0]],
            set_CoMy_range_index[slice_dict[im][1]],
            abs(slice_dict[im][1]-slice_dict[im][0])+1,
            endpoint = True)
        support_preds_across_mice.loc[CoMy_indices, im] = support_preds_sub
        
    support_preds_across_mice_dict[dict_lbl] = support_preds_across_mice
    
support_preds_across_mice_dict['incline_sum'] = support_preds_across_mice_dict['incline_sBA'] +\
                                    support_preds_across_mice_dict['incline_incline']  



#-------------PLOT!!-------------------                                
fig, ax = plt.subplots(1,1, figsize = (1.45,1.35)) 
xlims = (-0.1,-0.35)

if '3' in outcome_variable:
    yticks = [-0.1, 0, 0.1] 
    ylims=(-0.1, 0.1)
elif '2' in outcome_variable:
    yticks = [-0.1, 0, 0.1] 
    ylims=(-0.1, 0.1)
elif '4' in outcome_variable:
    yticks = [-0.05, 0, 0.05, 0.10, 0.1] 
    ylims=(-0.05, 0.1)
else:
    raise ValueError("ylim not specified!")

comy_pvals = []
for i, (clr,lbl, lnst,dim) in enumerate(zip(
        [FigConfig.colour_config['homolateral'][2], FigConfig.colour_config['greys'][0]],
        ["slope", "head h."],
        ['solid', 'dashed'],
        ['incline_sum', 'hh_sBA']
        )):   
    
    #-------------COMPUTE SHIFTS---------------------------
    dim0 = dim.split("_")[0]
    support_preds_across_mice_dict[f'{dim0}_shifts'] = support_preds_across_mice_dict[dim]
    
    # to align mice at zero, I should subtract the value corresponding to the
    # CoM that's common for all mice
    row_first_nonan = np.where(pd.isna(support_preds_across_mice).sum(axis=1)==0)[0][0]
    row_last_nonan = np.where(pd.isna(support_preds_across_mice).sum(axis=1)==0)[0][-1]
    support_preds_across_mice_dict[f'{dim0}_shifts'] = support_preds_across_mice_dict[dim]-support_preds_across_mice_dict[dim].iloc[row_first_nonan, :]
    support_preds_across_mice_dict[f'{dim0}_shifts'].iloc[:row_first_nonan,:] = np.nan
    support_preds_across_mice_dict[f'{dim0}_shifts'].iloc[row_last_nonan:,:] = np.nan
    
   
    #-------------COMPUTE SHIFTS---------------------------
    
    med = np.mean(support_preds_across_mice_dict[f'{dim0}_shifts'], 
                                axis=1)
    std = np.std(support_preds_across_mice_dict[f'{dim0}_shifts'],
                                axis=1)
    
    ax.hlines(ylims[1]-0.18*(ylims[1]-ylims[0]), -0.295+0.13*i, -0.23+0.13*i, color = clr, ls = lnst, lw = 1)
    ax.text(xlims[0] + (0.55 * (xlims[1]-xlims[0])) + 0.13*i,
            ylims[1] - (0.16* (ylims[1]-ylims[0])),
            lbl,
            color=clr,
            fontsize=5)
    
    ax.fill_between(set_CoMy_range_index,
                    med+std,#-med_first_nonan, 
                    med-std,#-med_first_nonan,
                    alpha = 0.2, 
                    facecolor = clr)
    ax.plot(set_CoMy_range_index,
            med,#-med_first_nonan, 
            color = clr,
            ls=lnst,
            lw = 1.5,
            label = lbl)
    
    # #-----------STATS: IS PC3-vs-COS SLOPE DIFFERENT FROM ZERO?-------
    # from scipy.stats import linregress
    # comy_stats = pd.DataFrame(zip(set_CoMy_range_index, med)).dropna()
    # _, _, _, p_value, _ = linregress(comy_stats.iloc[:,0], comy_stats.iloc[:,1])
    # comy_pvals.append(p_value)
    # print(f"Is slope significantly different from zero? {p_value}")

#-------------SAVE DICTS OR ADD STATS-------------------   
# appdx = "_hh_v_inc_"
# cols_of_interest = ['hh_shifts', 'incline_shifts']
# c_str = "_".join(cols_of_interest)

mod_predictors = ['speed', 'snoutBodyAngle', 'incline']
slopes_str = "".join(['pred2', 'pred3'])
mod_path = os.path.join(Config.paths['passiveOpto_output_folder'], 
   f"{yyyymmdd}_contCoefficients_mixedEffectsModel_linear_{outcome_variable}_vs_{mod_predictors[0]}_{mod_predictors[1]}_{mod_predictors[2]}_trialType_randSlopes{slopes_str}_interactionTRUEthreeway.csv")
# input_path = os.path.join(Config.paths['passiveOpto_output_folder'], 
#                         f"{yyyymmdd}_passiveOpto_phase{ph_str}_v_CoMy_stacked{appdx}ref_{ref}.csv")
if os.path.exists(mod_path):
    stats = pd.read_csv(mod_path, index_col = 0)
    p1 = np.min((stats.loc['pred2_centred', "Pr(>|t|)"], stats.loc['pred3_centred', "Pr(>|t|)"]))
    p2 = stats.loc["trialTypeslope", "Pr(>|t|)"]
    for i, p in enumerate([p1, p2]):
        ptext = ""
        if (p < np.asarray(FigConfig.p_thresholds)).sum() == 0:
            ptext += "n.s."   
        else:
            ptext += '*' * (p < np.asarray(FigConfig.p_thresholds)).sum()
    
        if i==1:
            ax.text(xlims[0] + (0.42 * (xlims[1]-xlims[0])),
                    ylims[1] - (0.26* (ylims[1]-ylims[0])),
                    f"vs          trials:\n{ptext}",
                    fontsize=5)
        else:
            ax.text(xlims[0] + (0.06 * (xlims[1]-xlims[0])),
                    ylims[1] - (0.05* (ylims[1]-ylims[0])),
                    f"centre of support: {ptext}",
                    fontsize=5)
    
#-------------SAVE DICTS OR ADD STATS-------------------  

ax.set_xlim(xlims[0], xlims[1]) 
ax.set_xticks([-0.1,-0.2,-0.3]) 
    
ax.set_ylim(ylims[0], ylims[1])
ax.set_yticks(yticks)
ax.set_ylabel(f"Shift in {ytlt}")   

ax.set_xlabel("AP centre of support\n(cm)")   
 
plt.tight_layout()
    
figtitle = f"passiveOpto_limbSupportPC{outcome_variable[-1]}_vs_CoS_incline_headheight.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
