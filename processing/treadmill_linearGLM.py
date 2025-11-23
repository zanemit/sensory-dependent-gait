import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import scipy.stats

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

import scipy.stats
from processing import data_loader, utils_math, utils_processing
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

def get_linear_slopes(
        predictors,
        yyyymmdd,
        outcome_variable,
        categ_var = 'refLimb',      # should be None if neither refLimb nor trialType; only one categ var possible
        interaction = 'TRUE',
        appdx = '_egr3', #appdx='_incline_COMBINEDtrialType'
        slopes = ['pred2'],
        outputDir = Config.paths['passiveOpto_output_folder'],
        mice = Config.passiveOpto_config['egr3_mice'],
        x_pred_range = None,
        special_other_predictors = {},
        merged = True,
        ref='lH1'
                ):
    """
    predictors (list of strs)
    yyyymmdd (str) : yyyy-mm-dd
    appdx (str) : '' or '_egr3' or 'egr3ctrl'
    slopes (list of strs) : 'pred1', 'pred2', 'pred1pred2' etc
    x_pred_range (dict or None) : 
        if None, pred_range has 100 values ranging from min/max predictor 
                value across the entire data; applied to ALL mice
        if dict {predictor: 1D array}, this array is used as pred_range for that
                predictor and ALL mice; can be of any length
                good if want to apply mouse-specific min/max range and average 
                across mice afterwards! Then this func muct be applied to each 
                mouse separately and averaged externally! Even though it returns data
                for all mice each time, it does not align predictor vals across mice
        if list (empty or full does not matter!), min/max predictor range is 
                computed for each mouse separately, but all arrays have 100 vals,
                so cannot be aligned across mice
                good for plotting individual mouse data w
    """
  
    # assign str depending on refLimb being a predictor
    rfl_str = f"_{categ_var}" if categ_var != None else ""
    
    dstr = 'strideParamsMerged' if merged else 'strideParams'
    # load full data
    datafull = data_loader.load_processed_data(dataToLoad = dstr,
                                               outputDir = outputDir,
                                               yyyymmdd = yyyymmdd,
                                               limb = ref, 
                                               appdx = appdx)[0]
   
    if 'headLVL' in datafull.columns:
        if 'deg' in datafull['headLVL'].iloc[-1] or 'deg' in datafull['headLVL'][0]:
            datafull['incline'] = [-int(x[3:]) for x in datafull['headLVL']]

    if 'trialType' in datafull.columns:
        if 'deg' in datafull['trialType'].iloc[-1] or 'deg' in datafull['trialType'][0]:
            datafull['incline'] = [-int(x[3:]) for x in datafull['trialType']]
            
    rh_str = appdx.split('_')[-1]+'_' if 'rH' in appdx else ''
    
    # subsetting the data into head height or slope trials only
    if 'COMBINEDtrialType' in appdx and categ_var!='trialType':
        if 'incline' in predictors:
           datafull = datafull[datafull['trialType']=='slope'].copy().reset_index(drop=True)
        else:
            datafull = datafull[datafull['trialType']=='headHeight'].copy().reset_index(drop=True)
    
    # load model coef files depending on the number of predictors & presence of slopes       
    if len(predictors)>2:
        contpath = os.path.join(outputDir, f"{yyyymmdd}_contCoefficients_mixedEffectsModel_linear_{rh_str}{outcome_variable}_vs_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_randSlopes{''.join(slopes)}_interaction{interaction}.csv")
        randpath = os.path.join(outputDir, f"{yyyymmdd}_randCoefficients_mixedEffectsModel_linear_{rh_str}{outcome_variable}_vs_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_randSlopes{''.join(slopes)}_interaction{interaction}.csv")
    else:
        contpath = os.path.join(outputDir, f"{yyyymmdd}_contCoefficients_mixedEffectsModel_linear_{rh_str}{outcome_variable}_vs_{predictors[0]}_{predictors[1]}{rfl_str}_randSlopes{''.join(slopes)}_interaction{interaction}.csv")
        randpath = os.path.join(outputDir, f"{yyyymmdd}_randCoefficients_mixedEffectsModel_linear_{rh_str}{outcome_variable}_vs_{predictors[0]}_{predictors[1]}{rfl_str}_randSlopes{''.join(slopes)}_interaction{interaction}.csv")
   
    contCoefficients = pd.read_csv(contpath, index_col=0)
    randCoefficients = pd.read_csv(randpath, index_col=0)

    # add "_centred" to slopes now that the files have been loaded
    if len(slopes)>0:
        slopes = [f"{s}_centred" for s in slopes]
        
    if np.all(x_pred_range) == None:
        pred_x_num = 100 
    else:
        pred_x_num = np.max([len(x) for x in x_pred_range.values()])
    
    if ref == 'COMBINED' and categ_var=='refLimb':
        ref_iterables = ['', 'rH1']
    elif categ_var=='trialType':
        ref_iterables = ['', f'{categ_var}slope']
    else:
        ref_iterables = ['']
    
    # define the data storage arrays
    support_preds_across_mice = np.empty((pred_x_num,
                                         len(predictors),
                                         len(mice)+1,
                                         len(ref_iterables)
                                         )) *np.nan
    
    x_range = np.empty((pred_x_num,
                             len(predictors),
                             )) *np.nan

    for b, predictor in enumerate(predictors):

        # define the predictor
        pred_id = np.where(np.asarray(predictors) == predictor)[0][0]   # refer to the `predictors` list
        pred = f'pred{pred_id+1}_centred' 
        
        # define the range of snout-hump angles or inclines (x axis predictor)
        pred_relevant = utils_processing.remove_outliers(datafull[predictor]) if predictor != 'incline' else np.asarray(datafull[predictor])
        pred_scaled = (pred_relevant - np.nanmean(pred_relevant))/np.nanstd(pred_relevant) 

        if np.all(x_pred_range) == None or predictor not in x_pred_range.keys():
            pred_range = np.linspace(pred_scaled.min(), pred_scaled.max(), num = pred_x_num)
        else:
            pred_range = (x_pred_range[predictor] - np.nanmean(pred_relevant))/np.nanstd(pred_relevant) 

        x_range[:, b] = (pred_range*np.nanstd(pred_relevant)) + np.nanmean(pred_relevant)
        
        # identify other predictors
        others = np.setdiff1d(predictors, predictor)
        nonpreds = []
        
        # find median param (excluding outliers) for the other predictors
        # if an other predictor is a key of special_other_predictors, use the provided prcnt value
        nonpred_dict = {}
        # nonpred_max_dict = {}
        for other in others:
            prcnt = 50 if other not in special_other_predictors.keys() else special_other_predictors[other]
            pred_id = np.where(np.asarray(predictors) == other)[0][0]
            nonpreds.append(f'pred{pred_id+1}_centred') 
            if other != 'incline':
                nonpred_relevant = utils_processing.remove_outliers(datafull[other])
            else:
                nonpred_relevant = datafull[other]
            nonpred_prct = (np.percentile(nonpred_relevant, prcnt) - np.nanmean(nonpred_relevant))/np.nanstd(nonpred_relevant)
            nonpred_dict[f'pred{pred_id+1}_centred'] = nonpred_prct  
        
        for i, refLimb in enumerate(ref_iterables):     
            output = contCoefficients.loc["(Intercept)", "Estimate"] +\
                    contCoefficients.loc[pred, "Estimate"] * pred_range 
                    
            if len(predictors)>1:
                output += contCoefficients.loc[nonpreds[0], "Estimate"] * nonpred_dict[nonpreds[0]]

            if len(predictors)>2:
                output += contCoefficients.loc[nonpreds[1], "Estimate"] * nonpred_dict[nonpreds[1]]
                
            if len(predictors)>3:
                output += contCoefficients.loc[nonpreds[2], "Estimate"] * nonpred_dict[nonpreds[2]] 
            
            if len(predictors)>4:
                raise ValueError("More than 4 predictors have not been implemented!")
       
            if (interaction == 'TRUE' or  interaction == 'TRUEthreeway' or interaction == 'TRUEfourway') and len(predictors)>=2:
                if pred in ['pred1_centred', 'pred2_centred']:
                    nonpred = np.setdiff1d(['pred1_centred', 'pred2_centred'], pred)[0]
                    output += contCoefficients.loc[f"pred1_centred:pred2_centred", "Estimate"] *\
                                pred_range *\
                                nonpred_dict[nonpred]
                else:
                    output += contCoefficients.loc[f"pred1_centred:pred2_centred", "Estimate"] *\
                              nonpred_dict['pred1_centred'] *\
                              nonpred_dict['pred2_centred']
                
            if (interaction == 'TRUEsecondary' or interaction == 'TRUEthreeway' or interaction == 'TRUEfourway') and len(predictors)>=3:
                if pred in ['pred2_centred', 'pred3_centred']:
                    nonpred = np.setdiff1d(['pred2_centred', 'pred3_centred'], pred)[0]
                    output += contCoefficients.loc[f"pred2_centred:pred3_centred", "Estimate"] *\
                                pred_range *\
                                nonpred_dict[nonpred]
                else:
                    output += contCoefficients.loc[f"pred2_centred:pred3_centred", "Estimate"] *\
                              nonpred_dict['pred2_centred'] *\
                              nonpred_dict['pred3_centred']
                
            if (interaction == 'TRUEthreeway' or interaction == 'TRUEfourway') and len(predictors)>=3:
                if pred in ['pred1_centred', 'pred2_centred', 'pred3_centred']:
                    nonpredlist = np.setdiff1d(['pred1_centred', 'pred2_centred', 'pred3_centred'], pred)
                    if pred in ['pred1_centred', 'pred3_centred']:
                        nonpred = np.setdiff1d(['pred1_centred', 'pred3_centred'], pred)[0]
                        output += contCoefficients.loc[f"pred1_centred:pred3_centred", "Estimate"] *\
                                    pred_range *\
                                    nonpred_dict[nonpred]
                    else:
                        output += contCoefficients.loc[f"pred1_centred:pred3_centred", "Estimate"] *\
                                  nonpred_dict['pred1_centred'] *\
                                  nonpred_dict['pred3_centred']
                    
                    output += contCoefficients.loc[f"pred1_centred:pred2_centred:pred3_centred", "Estimate"] *\
                             pred_range *\
                             nonpred_dict[nonpredlist[0]] *\
                             nonpred_dict[nonpredlist[1]]
                    
                else:
                    output += contCoefficients.loc[f"pred1_centred:pred2_centred:pred3_centred", "Estimate"] *\
                              nonpred_dict['pred1_centred'] *\
                              nonpred_dict['pred2_centred'] *\
                              nonpred_dict['pred3_centred']
                              
                          
            if interaction == 'TRUEfourway' and len(predictors)>=4:
                if pred in ['pred1_centred', 'pred2_centred', 'pred4_centred']:
                    nonpredlist = np.setdiff1d(['pred1_centred', 'pred2_centred', 'pred4_centred'], pred)
                    if pred in ['pred1_centred', 'pred4_centred']:
                        nonpred = np.setdiff1d(['pred1_centred', 'pred4_centred'], pred)[0]
                        output += contCoefficients.loc[f"pred1_centred:pred4_centred", "Estimate"] *\
                                    pred_range *\
                                    nonpred_dict[nonpred]
                    else:
                        output += contCoefficients.loc[f"pred1_centred:pred4_centred", "Estimate"] *\
                                  nonpred_dict['pred1_centred'] *\
                                  nonpred_dict['pred4_centred']
                                  
                    if pred in ['pred2_centred', 'pred4_centred']:
                        nonpred = np.setdiff1d(['pred2_centred', 'pred4_centred'], pred)[0]
                        output += contCoefficients.loc[f"pred2_centred:pred4_centred", "Estimate"] *\
                                    pred_range *\
                                    nonpred_dict[nonpred]
                    else:
                        output += contCoefficients.loc[f"pred2_centred:pred4_centred", "Estimate"] *\
                                  nonpred_dict['pred2_centred'] *\
                                  nonpred_dict['pred4_centred']
                                  
                    output += contCoefficients.loc[f"pred1_centred:pred2_centred:pred4_centred", "Estimate"] *\
                              pred_range *\
                              nonpred_dict[nonpredlist[0]] *\
                              nonpred_dict[nonpredlist[1]]
                else:
                    output += contCoefficients.loc[f"pred1_centred:pred2_centred:pred4_centred", "Estimate"] *\
                              nonpred_dict['pred1_centred'] *\
                              nonpred_dict['pred2_centred'] *\
                              nonpred_dict['pred4_centred']
                              
                if pred in ['pred1_centred', 'pred3_centred', 'pred4_centred']:
                    nonpredlist = np.setdiff1d(['pred1_centred', 'pred3_centred', 'pred4_centred'], pred)
                    if pred in ['pred3_centred', 'pred4_centred']:
                        nonpred = np.setdiff1d(['pred3_centred', 'pred4_centred'], pred)[0]
                        output += contCoefficients.loc[f"pred3_centred:pred4_centred", "Estimate"] *\
                                    pred_range *\
                                    nonpred_dict[nonpred]
                    else:
                        output += contCoefficients.loc[f"pred3_centred:pred4_centred", "Estimate"] *\
                                  nonpred_dict['pred3_centred'] *\
                                  nonpred_dict['pred4_centred']
                                  
                    output += contCoefficients.loc[f"pred1_centred:pred3_centred:pred4_centred", "Estimate"] *\
                              pred_range *\
                              nonpred_dict[nonpredlist[0]] *\
                              nonpred_dict[nonpredlist[1]]
                else:
                    output += contCoefficients.loc[f"pred1_centred:pred3_centred:pred4_centred", "Estimate"] *\
                              nonpred_dict['pred1_centred'] *\
                              nonpred_dict['pred3_centred'] *\
                              nonpred_dict['pred4_centred']
                              
                if pred in ['pred2_centred', 'pred3_centred', 'pred4_centred']:
                    nonpredlist = np.setdiff1d(['pred2_centred', 'pred3_centred', 'pred4_centred'], pred)
                    output += contCoefficients.loc[f"pred2_centred:pred3_centred:pred4_centred", "Estimate"] *\
                              pred_range *\
                              nonpred_dict[nonpredlist[0]] *\
                              nonpred_dict[nonpredlist[1]]           
                else:
                    output += contCoefficients.loc[f"pred2_centred:pred3_centred:pred4_centred", "Estimate"] *\
                              nonpred_dict['pred2_centred'] *\
                              nonpred_dict['pred3_centred'] *\
                              nonpred_dict['pred4_centred']
                
                output += contCoefficients.loc[f"pred1_centred:pred2_centred:pred3_centred:pred4_centred", "Estimate"] *\
                          pred_range *\
                          nonpred_dict[nonpreds[0]] *\
                          nonpred_dict[nonpreds[1]] *\
                          nonpred_dict[nonpreds[2]]    
                

            if refLimb == ref_iterables[-1] and len(ref_iterables)>1:
                output += contCoefficients.loc[f"{refLimb}", "Estimate"]
                
                # what if refLimb is part of an interaction?
                if interaction=='TRUEfourway' and len(predictors)<4:
                    output += contCoefficients.loc[f"{pred}:{refLimb}", "Estimate"] * pred_range
                    
                    nonpredlist = np.setdiff1d(['pred1_centred', 'pred2_centred', 'pred3_centred'], pred)
                    for nonpred in nonpredlist:
                        output += contCoefficients.loc[f"{nonpred}:{refLimb}", "Estimate"] * nonpred_dict[nonpred]
                    if pred in ['pred1_centred', 'pred2_centred']:
                        nonpred = np.setdiff1d(['pred1_centred', 'pred2_centred'], pred)[0]
                        output += contCoefficients.loc[f"{pred}:pred3_centred:{refLimb}", "Estimate"]*\
                                    pred_range *\
                                    nonpred_dict['pred3_centred']
                        output += contCoefficients.loc[f"{nonpred}:pred3_centred:{refLimb}", "Estimate"]*\
                                    nonpred_dict[nonpred]  *\
                                    nonpred_dict['pred3_centred']
                        output += contCoefficients.loc[f"pred1_centred:pred2_centred:{refLimb}", "Estimate"]*\
                                    pred_range *\
                                    nonpred_dict[nonpred] 
                    else: # pred=='pred3_centred'
                        output += contCoefficients.loc[f"pred1_centred:pred3_centred:{refLimb}", "Estimate"]*\
                                     pred_range *\
                                     nonpred_dict['pred1_centred']
                        output += contCoefficients.loc[f"pred2_centred:pred3_centred:{refLimb}", "Estimate"]*\
                                    pred_range *\
                                    nonpred_dict['pred2_centred']
                        output += contCoefficients.loc[f"pred1_centred:pred2_centred:{refLimb}", "Estimate"]*\
                                    nonpred_dict['pred1_centred'] *\
                                    nonpred_dict['pred2_centred']
                                    
                    output += contCoefficients.loc[f"pred1_centred:pred2_centred:pred3_centred:{refLimb}", "Estimate"] *\
                                pred_range *\
                                nonpred_dict[nonpredlist[0]] *\
                                nonpred_dict[nonpredlist[1]]
                    
                if interaction=='TRUEthreeway' and len(predictors)<3: # assume that refLimb is part of the interaction
                    output += contCoefficients.loc[f"{pred}:{refLimb}", "Estimate"] * pred_range
                    
                    nonpred = np.setdiff1d(['pred1_centred', 'pred2_centred'], pred)[0]
                    output += contCoefficients.loc[f"{nonpred}:{refLimb}", "Estimate"] * nonpred_dict[nonpred]
                    output += contCoefficients.loc[f"pred1_centred:pred2_centred:{refLimb}", "Estimate"] *\
                                pred_range *\
                                nonpred_dict[nonpred]
            
            if len(slopes)>0:
                support_preds_across_mice[:, b, 0, i] = (output * np.nanstd(datafull[outcome_variable])) + np.nanmean(datafull[outcome_variable])
                
                for im, m in enumerate(mice):
                    # add random intercepts
                    output_m = output + randCoefficients.loc[m, "(Intercept)"]
                    
                    # find out which parameter to add the slope t
                    # is it the predictor?
                    if pred in slopes:
                        output_m += randCoefficients.loc[m, pred] * pred_range                        
                        
                    nonpred_slopes = np.intersect1d(slopes, list(nonpred_dict.keys()))
                    if len(nonpred_slopes)>0:
                        for s in nonpred_slopes:
                            output_m += randCoefficients.loc[m, s] * nonpred_dict[s]  
                    support_preds_across_mice[:, b, im+1, i] = (output_m * np.nanstd(datafull[outcome_variable])) + np.nanmean(datafull[outcome_variable])
                    
            else:
                support_prediction = (output * np.nanstd(datafull[outcome_variable])) + np.nanmean(datafull[outcome_variable])
                support_preds_across_mice[:, b, :, i] = np.repeat(support_prediction[:, np.newaxis], 
                                                    len(mice)+1,
                                                    axis = 2)

    return (x_range,
            support_preds_across_mice)

def get_predictor_range(predictor):
    """
    predictor (str) can be 'snoutBodyAngle', 'incline', 'duty_ref', 'speed', 'headHW'
    """
    if predictor == 'incline' : 
        xlim = (-41,45)
        xticks = [-40,0,40]
        xlabel = 'Incline (deg)'
    elif predictor == 'snoutBodyAngle':
        xlim = (139,181)
        xticks = [140,150,160,170,180]
        xlabel = 'Snout-hump angle (deg)'
    elif predictor == 'duty_ratio' or predictor == 'dutyratio':
        xlim = (0.3,2.1)
        xticks = [0.3,0.9,1.5,2.1]
        xlabel = 'Hind-fore duty factor ratio'
    elif 'duty' in predictor:
        xlim = (0.39,1.15)
        xticks = [0.4,0.6,0.8,1]
        lbl2 = predictor.split("_")[-1]
        xlabel = f'Duty factor - {lbl2}'
    elif predictor == 'headHW':
        xlim = (0,1.3)
        xticks = [0,0.4,0.8,1.2]
        xlabel = 'Weight-adjusted\nhead height (a.u.)'
    elif predictor == 'speed':
        xlim = (0,150)
        xticks = [0,30,60,90,120,150]
        xlabel = 'Speed (cm/s)'
    elif predictor == 'weight':
        xlim = (17,32)
        xticks = [17,20,23,26,29,32]
        xlabel = 'Weight (g)'
    elif predictor == 'strideLength':
        xlim = (0,6)
        xticks = [0,2,4,6]
        xlabel = 'Stride length (??)'
    elif predictor == 'sba_headHW_residuals':
        xlim = (-30,30)
        xticks = [-30,0,30]
        xlabel = 'Snout-hump angle\nresiduals (deg)'
    elif 'limbSupportPC' in predictor:
        xlim = (-2,2)
        xticks = [-2,0,2]
        xlabel = 'Limb support PC1'
    elif predictor == 'frac2hmlt':
        xlim = (0,1)
        xticks = [0,0.01,0.02]
        xlabel = 'Fraction ogf homolateral support'
    else:
        raise ValueError("Invalid predictor supplied! This functionality needs to be added!")
    return (xlim, xticks, xlabel)
    