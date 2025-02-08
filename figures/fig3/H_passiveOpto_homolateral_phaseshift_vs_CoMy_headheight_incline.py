import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os 
from matplotlib import pyplot as plt
import scipy.stats

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

import scipy.stats
from processing import data_loader, utils_math, utils_processing, treadmill_circGLM
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

# limb = 'homolateral0'
# ref = 'COMBINED'
# sample_nums = [12373, 12373, 12510]
# datafracs = [0.4, 0.4, 0.4]
# catvar = "refLimb"

#-----------testing if there is a difference between per-mouse v slope model----
#-----------or if the discrepancy is due to the use of lH1 or COMB as reflimbs--
limb = 'lF0' # should change
ref = 'lH1'
datafracs = [0.7, 0.7, 0.6]
sample_nums = [13684, 13684, 13110]
catvar = None

#-----------testing if there is a difference between per-mouse v slope model----
#-----------or if the discrepancy is due to the use of lH1 or COMB as reflimbs--
# limb = 'rF0' # should change
# ref = 'rH1'
# datafracs = [0.7, 0.7, 0.6]
# sample_nums = [13507, 13507, 13060]
# catvar = [False, False, False]

#----------------------------------------------------------------

yyyymmdd = '2022-08-18'
mouselist = Config.passiveOpto_config['mice']
ph_str = "shift"

set_CoMy_ant = 0.5
set_CoMy_post = -1
set_CoMy_range_index = np.linspace(set_CoMy_ant,
                          set_CoMy_post,
                          num = abs(int(((set_CoMy_ant-set_CoMy_post)*64)+1)), 
                          endpoint=True)
set_CoMy_range_index_str = set_CoMy_range_index*Config.forceplate_config["fore_hind_post_cm"]/2

phase_preds_across_mice_dict = {}
slice_dict = {}
for i, (yyyymmdd_fp, appdx, predictorlist, slopes, predictor, interaction, sample_num, data_frac, dict_lbl) in enumerate(zip(
           ['2021-10-26', '2022-04-04', '2021-10-26'],
           ['_incline', '_incline', ''],
           [['speed', 'snoutBodyAngle', 'incline'], ['speed', 'snoutBodyAngle', 'incline'], ['speed', 'snoutBodyAngle']],
           [['pred2', 'pred3'], ['pred2', 'pred3'], ['pred2']],
           ['snoutBodyAngle', 'levels', 'snoutBodyAngle'],
           ['TRUEsecondary', 'TRUEsecondary', 'TRUE'],
           sample_nums, 
           datafracs, 
           ['incline_sBA', 'incline_incline', 'hh_sBA'],
        )):  
    
    phase_preds_across_mice = pd.DataFrame(np.empty((set_CoMy_range_index.shape[0], 
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
    else:
        predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
        predictor_trdm = predictor
        
    # CoMy vs snout-hump angle
    if yyyymmdd_fp=='2022-04-04':
        CoMy_pred_path = os.path.join(
            Config.paths["forceplate_output_folder"],"old_incline",f"{yyyymmdd_fp}_mixedEffectsModel_linear_COMy_{predictor}.csv"
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
                                 phase_preds_across_mice.shape[0], 
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
        
        x_preds, phase_preds = treadmill_circGLM.get_circGLM_slopes(
                predictors = predictorlist,
                yyyymmdd = yyyymmdd,
                limb = limb,
                ref = ref,
                samples = sample_num,
                interaction = interaction,
                appdx = appdx,
                datafrac = data_frac,
                categ_var=catvar,
                slopes = slopes,
                outputDir = Config.paths['passiveOpto_output_folder'],
                iterations = 1000,
                mice = mouselist,
                x_pred_range = {predictor_trdm: x_target_range}# - np.nanmean(datafull_sub[predictor_trdm])}
                        ) 
        
        phase_preds_sub = phase_preds[:, :, predictor_id, im+1, 0]
        
        mean_phase = scipy.stats.circmean(phase_preds_sub, 
                                          high = np.pi, 
                                          low = -np.pi, 
                                          axis=0)
        if np.any(abs(np.diff(mean_phase))>5):
            phase_preds_sub[phase_preds_sub<0] = phase_preds_sub[phase_preds_sub<0]+2*np.pi
            mean_phase = scipy.stats.circmean(phase_preds_sub, 
                                              high = 2*np.pi, 
                                              low = 0, 
                                              axis=0)
        
        CoMy_indices = np.linspace(
            set_CoMy_range_index[slice_dict[im][0]],
            set_CoMy_range_index[slice_dict[im][1]],
            abs(slice_dict[im][1]-slice_dict[im][0])+1,
            endpoint = True)
        phase_preds_across_mice.loc[CoMy_indices, im] = mean_phase
        
    phase_preds_across_mice_dict[dict_lbl] = phase_preds_across_mice
    
phase_preds_across_mice_dict['incline_sum'] = phase_preds_across_mice_dict['incline_sBA'] +\
                                    phase_preds_across_mice_dict['incline_incline']  


#-------------PLOT!!-------------------                                
fig, ax = plt.subplots(1,1, figsize = (1.5,1.35)) 
xlims = (-0.1,-0.4)
yticks = [-0.5*np.pi, 0, 0.5*np.pi] 
ylims=(yticks[0]-(0.1*np.pi), yticks[-1])
    
for i, (clr,lbl, lnst,dim) in enumerate(zip(
        [FigConfig.colour_config['homolateral'][2], FigConfig.colour_config['greys'][0]],
        ["slope", "head h."],
        ['solid', 'dashed'],
        ['incline_sum', 'hh_sBA']
        )):   
    
    #-------------COMPUTE SHIFTS---------------------------
    dim0 = dim.split("_")[0]
    phase_preds_across_mice_dict[f'{dim0}_shifts'] = phase_preds_across_mice_dict[dim]
    
    # to align mice at zero, I should subtract the value corresponding to the
    # CoM that's common for all mice
    row_first_nonan = np.where(pd.isna(phase_preds_across_mice).sum(axis=1)==0)[0][0]
    phase_preds_across_mice_dict[f'{dim0}_shifts'] = phase_preds_across_mice_dict[dim]-phase_preds_across_mice_dict[dim].iloc[row_first_nonan, :]
    phase_preds_across_mice_dict[f'{dim0}_shifts'].iloc[:row_first_nonan,:] = np.nan
    
    # for im, m in enumerate(mouselist):
    #     col = phase_preds_across_mice_dict[dim].iloc[:,im]
        
    #     # subtract the first non-nan value, but then mice are not aligned at zero
    #     col_first_nonan = col[~np.isnan(col)].iloc[0]
    #     phase_preds_across_mice_dict[f'{dim0}_shifts'].loc[:, im] = np.asarray(col)-col_first_nonan
    
        
        
    
    #-------------COMPUTE SHIFTS---------------------------
    
    med = scipy.stats.circmean(phase_preds_across_mice_dict[f'{dim0}_shifts'], 
                                high = np.pi, 
                                low = -np.pi,
                                axis=1)
    std = scipy.stats.circstd(phase_preds_across_mice_dict[f'{dim0}_shifts'],
                                high = np.pi, 
                                low = -np.pi,
                                axis=1)
    
    ax.hlines(ylims[1]-0.3, -0.33+0.15*i, -0.265+0.15*i, color = clr, ls = lnst, lw = 1)
    ax.text(xlims[0] + (0.525 * (xlims[1]-xlims[0])) + 0.15*i,
            ylims[1] - (0.05* (ylims[1]-ylims[0])),
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

#-------------SAVE DICTS OR ADD STATS-------------------   
appdx = "_hh_v_inc_"
cols_of_interest = ['hh_shifts', 'incline_shifts']
c_str = "_".join(cols_of_interest)

mod_path = os.path.join(Config.paths['passiveOpto_output_folder'], 
                        f"{yyyymmdd}_MixedEffectsModel_passiveOpto_phase{ph_str}_v_CoMy_stacked{appdx}{c_str}_ref_{ref}.csv")
input_path = os.path.join(Config.paths['passiveOpto_output_folder'], 
                        f"{yyyymmdd}_passiveOpto_phase{ph_str}_v_CoMy_stacked{appdx}ref_{ref}.csv")
if os.path.exists(mod_path):
    stats = pd.read_csv(mod_path, index_col = 0)
    p = stats.loc["conditionsincline_shifts", "Pr(>|t|)"]
    ptext = ""
    if (p < np.asarray(FigConfig.p_thresholds)).sum() == 0:
        ptext += "n.s."   
    else:
        ptext += '*' * (p < np.asarray(FigConfig.p_thresholds)).sum()
    
    p_x = np.where(~np.isnan(med))[0][-1]+3
    p_y = med[~np.isnan(med)][-1]
    
    # ax.text(set_CoMy_range_index[p_x],
    #         p_y,
    #         ptext, 
    #         ha = 'center')
    
    ax.text(xlims[0] + (0.39 * (xlims[1]-xlims[0])),
            ylims[1] - (0.15* (ylims[1]-ylims[0])),
            f"vs          trials:\n{ptext}",
            fontsize=5)
    
elif os.path.exists(input_path):
    print("""
          You should run the LMM using 'lmer_script_treadmill_phase_v_CoMy.R'! 
          The input file for the model exists, but stats cannot be printed without 
          the model itself!
          """)
else:
    print("""
          Generating and saving the input file... 
          Head over to R now!
          Run 'lmer_script_treadmill_phase_v_CoMy.R'
          """)  
    df_for_reg = utils_processing.transform_dict_into_df(phase_preds_across_mice_dict)
    df_for_reg = utils_processing.transform_multicolumn_df_for_regression(
                                df_for_reg, 
                                cols_to_keep = ['cln', 'idx']
                                )
    df_for_reg.to_csv(input_path)
    

#-------------SAVE DICTS OR ADD STATS-------------------  

ax.set_xlim(xlims[0], xlims[1]) 
ax.set_xticks([-0.1,-0.2,-0.3, -0.4]) 
    
ax.set_ylim(yticks[0]-(0.1*np.pi), yticks[-1])
ax.set_yticks(yticks)
ax.set_yticklabels([f"{num:.1f}π" if num not in [0,1] else "π" if num == 1 else "0" for num in np.asarray(yticks)/np.pi])
ax.set_ylabel(f"Homolateral phase\n{ph_str} (rad)")   

ax.set_xlabel("AP centre of support\n(cm)")   
 
plt.tight_layout()
    
figtitle = f"passiveOpto_{limb}_phase{ph_str}_vs_CoS_continuous_incline_headheight_different.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
