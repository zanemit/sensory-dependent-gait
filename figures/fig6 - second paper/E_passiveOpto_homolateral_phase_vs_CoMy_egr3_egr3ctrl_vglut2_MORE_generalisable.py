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
# samplenums = [11079, 12103, 12510]
# datafracs = [0.6, 0.3, 0.4]
# reflimb_strs = [True, True, True]

limb = 'lF0'
ref = 'lH1'
samplenums = [11584, 14667, 13110]
datafracs = [1, 0.6, 0.6]
reflimb_strs = [False, False, False]

#----------------------------------------------------------------
#----------------------------------------------------------------

predictorlist = ['speed', 'snoutBodyAngle']
predictor = 'snoutBodyAngle'
predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
slopes = ['pred2']
interaction = 'TRUE'

set_CoMy_ant = 0.5
set_CoMy_post = -1
set_CoMy_range_index = np.linspace(set_CoMy_ant,
                          set_CoMy_post,
                          num = abs(int(((set_CoMy_ant-set_CoMy_post)*32)+1)), 
                          endpoint=True)

input_path = os.path.join(Config.paths['passiveOpto_output_folder'], 
                         f"2023-08-14_passiveOpto_phase_v_CoMy_stackedref_egr3_egr3ctrl_vglut2_{ref}.csv")

cols_to_keep = ['speed', 'snoutBodyAngle', limb, 'mouseID'] 
stacked_datafull = pd.DataFrame(columns=cols_to_keep + ['condition', 'CoMy'])
phase_preds_across_mice_dict = {}
slice_dict = {}
for i, (yyyymmdd, yyyymmdd_fp, appdx, sample_num, data_frac, mouselist, dict_lbl, rfl_str) in enumerate(zip(
           ['2023-08-14', '2023-09-21', '2022-08-18'],
           ['2023-11-06', '2021-10-26', '2021-10-26'],
           ['_egr3', '_egr3ctrl', ''],
           samplenums,
           datafracs,
           [Config.passiveOpto_config['egr3_mice'], Config.passiveOpto_config['egr3_ctrl_mice'], Config.passiveOpto_config['mice']],
           ['egr3', 'egr3ctrl', 'vglut2'],
           reflimb_strs
           )):   
    datafull = data_loader.load_processed_data(dataToLoad = 'strideParamsMerged', 
                                                yyyymmdd = yyyymmdd, 
                                                limb = ref, 
                                                appdx = appdx)[0]
    
    phase_preds_across_mice = pd.DataFrame(np.empty((set_CoMy_range_index.shape[0], 
                                                     len(mouselist) 
                                                     ))*np.nan,
                                           index = set_CoMy_range_index)
     
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
    
    # compute the corresponding sBA range 
    get_pred_ant = ((set_CoMy_ant-np.nanmean(data_fp['CoMy_mean']) - CoMy_pred_intercept) / CoMy_pred_slope) + np.nanmean(data_fp['param']) 
    get_pred_post = ((set_CoMy_post-np.nanmean(data_fp['CoMy_mean']) - CoMy_pred_intercept) / CoMy_pred_slope) + np.nanmean(data_fp['param']) 
    
    set_pred_range = np.linspace(get_pred_ant, 
                                 get_pred_post, 
                                 phase_preds_across_mice.shape[0], 
                                 endpoint= True)
 
    # iterate over mice
    for im, m in enumerate(mouselist):
        # subset dataframe
        datafull_sub = datafull[datafull['mouseID'] == m]
        
        # set min/max predictor range that will be used for all mice
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
            slice_dict[im][1]-slice_dict[im][0]+1,
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
                categ_var = None,
                slopes = slopes,
                # merged=True,
                outputDir = Config.paths['passiveOpto_output_folder'],
                iterations = 1000,
                mice = mouselist,
                x_pred_range = {predictor: x_target_range }#- np.nanmean(datafull_sub['snoutBodyAngle'])}
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
            slice_dict[im][1]-slice_dict[im][0]+1,
            endpoint = True)
        phase_preds_across_mice.loc[CoMy_indices, im] = mean_phase
    phase_preds_across_mice_dict[dict_lbl] = phase_preds_across_mice
    
    if not os.path.exists(input_path):
        datafull_to_save = datafull[cols_to_keep]
        datafull_to_save.loc[:,'condition'] = dict_lbl
        datafull_to_save.loc[:, 'CoMy'] =  ((datafull_to_save.loc[:, 'snoutBodyAngle']-np.nanmean(data_fp['param']))* CoMy_pred_slope) +np.nanmean(data_fp['CoMy_mean']) 
        stacked_datafull = pd.concat((stacked_datafull, datafull_to_save),
                                     ignore_index=True)
        
#-------------PLOT-------------------                                   
fig, ax = plt.subplots(1,1, figsize = (1.4,1.35))
for b, (clr,lbl,lnst,dim) in enumerate(zip(
        [FigConfig.colour_config['homolateral'][2], FigConfig.colour_config['greys'][1], FigConfig.colour_config['greys'][1]],
        ['MS-deficient', 'Control (littermates)', 'Control (vGlut2-Cre)'],
        ['solid', 'solid', 'dashed'],
        ['egr3', 'egr3ctrl', 'vglut2']
        )):   
    med = scipy.stats.circmean(phase_preds_across_mice_dict[dim], 
                                high = 2*np.pi, 
                                low = 0,
                                axis=1)
    std = scipy.stats.circstd(phase_preds_across_mice_dict[dim],
                                high = 2*np.pi, 
                                low = 0,
                                axis=1)

    ax.fill_between(set_CoMy_range_index*Config.forceplate_config["fore_hind_post_cm"]/2,
                    med+std, 
                    med-std,
                    alpha = 0.2, 
                    facecolor = clr)
    ax.plot(set_CoMy_range_index*Config.forceplate_config["fore_hind_post_cm"]/2,
            med, 
            color = clr,
            linestyle = lnst,
            lw = 1.5,
            label = lbl)
    
    # TEXT
    if dim=='egr3ctrl':
        ax.text(0.65, 1.5*np.pi, "CTRL", color=clr, fontsize=5)
        ax.hlines(1.48*np.pi, 0.65, 0.2, ls=lnst, color=clr, lw=0.7)
        ax.text(0.65, 1.32*np.pi, "CTRL", color=clr, fontsize=5)
        ax.hlines(1.3*np.pi, 0.65, 0.2, ls=lnst, color=clr, lw=0.7)
        ax.text(0.15, 1.5*np.pi, "vs              :",  fontsize=5)
        ax.text(0.15, 1.32*np.pi, "vs           :",  fontsize=5)
    else:
        txt = "MSA def" if b==0 else "vGlut2"
        ax.text(-0.1, (1.5-(b*0.09))*np.pi, txt, color=clr, fontsize=5)
        ax.hlines((1.48-(b*0.09))*np.pi, -0.1, -0.53+(0.03*b), color=clr, ls=lnst, lw=0.7)
    
#-------------PLOT-------------------      
    
#-------------SAVE DICTS OR ADD STATS-------------------   

comparisons_of_interest = [('egr3', 'egr3ctrl'), ('egr3ctrl', 'vglut2')]
sample_fracs = [0.4, 0.35]
data_nums = [11840, 13341]
yyyymmdd = '2023-08-14'
num_comparisons = 2

for ic, (coi, sfrac, dnum) in enumerate(zip(
        comparisons_of_interest,
        sample_fracs,
        data_nums
        )):
    c_str = "_v_".join(coi)
    txt_clr = FigConfig.colour_config['homolateral'][2] if 'egr3' in coi else FigConfig.colour_config['greys'][1]
    mod_path = os.path.join(Config.paths['passiveOpto_output_folder'], 
                            f"{yyyymmdd}_coefCircContinuous_{limb}_ref{ref}_speed_snoutBodyAngle_condition_interactionTRUEthreeway_randMouseSLOPEpred2_{sfrac}data{dnum}s_1000its_100burn_3lag_{c_str}.csv")

    if os.path.exists(mod_path):
        stats = pd.read_csv(mod_path, index_col = 0)
        confint = stats.loc[f"pred2:condition_factor{coi[-1]} SSDO", ["LB HPD","UB HPD"]]
        is_same_sign = np.all(np.sign(confint)==np.sign(confint.iloc[0]))
        ptext = "*" if is_same_sign else "n.s."        
        ax.text(-0.98+(0.2*ic), (1.47-(ic*0.15))*np.pi, ptext, fontsize=5)
        

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
        stacked_datafull.to_csv(input_path, index=False)
        # df_for_reg = utils_processing.transform_dict_into_df(phase_preds_across_mice_dict)
        # df_for_reg = utils_processing.transform_multicolumn_df_for_regression(
        #                             df_for_reg, 
        #                             cols_to_keep = ['cln', 'idx']
        #                             )
        # df_for_reg.to_csv(input_path)
#-------------SAVE DICTS OR ADD STATS------------------- 


#-------------FORMAT THE PLOT------------------- 
ax.set_xlim(0.8, -1) 
ax.set_xticks([0.5,0,-0.5,-1], labels = ["0.5", "0", "-0.5", "-1"])
ax.set_ylim(0.4*np.pi, 1.5*np.pi)
ax.set_yticks([ 0.5*np.pi, np.pi, 1.5*np.pi])
ax.set_yticklabels([ "0.5π", "π", "1.5π"])        
        
ax.set_ylabel("Homolateral phase\n(rad)")   
ax.set_xlabel("AP centre of\nsupport (cm)")   
# lgd = fig.legend(bbox_to_anchor=(0.3,0.95,0.65,0.3), mode="expand", borderaxespad=0.1, fontsize = 6)
#-------------FORMAT THE PLOT------------------- 

plt.tight_layout()
    
figtitle = f"passiveOpto_{limb}_phase_vs_CoS_continuous_egr3_egr3ctrl_vglut2.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            # bbox_inches = 'tight',
            transparent = True)
