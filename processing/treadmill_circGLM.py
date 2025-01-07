import sys
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

import scipy.stats
from processing import data_loader, utils_math, utils_processing
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

def get_circGLM_slopes(
        predictors,
        yyyymmdd,
        limb,
        ref,
        categ_var = 'refLimb',      # should be None if neither refLimb nor trialType; only one categ var possible
        samples = 11079,
        interaction = 'TRUE',
        appdx = '_egr3',
        datafrac = 0.6,
        slopes = ['pred2'],
        outputDir = Config.paths['passiveOpto_output_folder'],
        iterations = 1000,
        mice = Config.passiveOpto_config['egr3_mice'],
        x_pred_range = None,
        special_other_predictors = {},
        sBA_split_str = 'sBAsplitFALSE',
        merged = False
                ):
    """
    predictors (list of strs)
    yyyymmdd (str) : yyyy-mm-dd
    limb (str) : 'homolateral0', 'lF0', ...
    ref (str) : 'COMBINED', 'lH1', ...
    samples (int) : number of samples used to fit the model
    interaction (str) : 'TRUE' or 'FALSE'
    appdx (str) : '' or '_egr3' or 'egr3ctrl'
    datafrac (float) : fraction of data used to fit the model,
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
    
    beta = fixed effects predictors
    b_pred = random effect intercept (assumed to always be present)
    """
    # assign slope-related appdx for file name
    slope_appdx = f'SLOPE{"".join(slopes)}' if len(slopes)>0 else ''
    
    # assign str depending on refLimb being a predictor
    rfl_str = f"_{categ_var}" if categ_var != None else ""
    
    dstr = 'strideParamsMerged' if merged else 'strideParams'
    # load full data    
    if categ_var == 'trialType':
        filenames = {'2021-10-23': {False: 'strideParams_COMBINEDtrialType'}, 
                     '2022-08-18': {True: 'strideParamsMerged_incline_COMBINEDtrialType_lH1', False: 'strideParams_incline_trialType_COMBINED'},
                     '2022-05-06': {True: 'strideParamsMerged_COMBINEDtrialType_lH1'}}
        datafull = pd.read_csv(Path(outputDir)/ f"{yyyymmdd}_{filenames[yyyymmdd][merged]}.csv")
    else:
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
    
    # load model coef files depending on the number of predictors & presence of slopes
    if len(predictors) == 5:
        beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_{predictors[4]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_{predictors[4]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b1pred_path = Path(outputDir) / f"{yyyymmdd}_b1_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_{predictors[4]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b2pred_path = Path(outputDir) / f"{yyyymmdd}_b2_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_{predictors[4]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        
        if len(slopes)>0:
            b1_dict = {};  b2_dict = {}
            for s in slopes:
                b1pred_k_path = Path(outputDir) / f"{yyyymmdd}_b1_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_{predictors[4]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b2pred_k_path = Path(outputDir) / f"{yyyymmdd}_b2_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_{predictors[4]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b1_dict[s] = pd.read_csv(b1pred_k_path, index_col = 0)
                b2_dict[s] = pd.read_csv(b2pred_k_path, index_col = 0)
                
    elif len(predictors) == 4:
        beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b1pred_path = Path(outputDir) / f"{yyyymmdd}_b1_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b2pred_path = Path(outputDir) / f"{yyyymmdd}_b2_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        
        if len(slopes)>0:
            b1_dict = {};  b2_dict = {}
            for s in slopes:
                b1pred_k_path = Path(outputDir) / f"{yyyymmdd}_b1_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b2pred_k_path = Path(outputDir) / f"{yyyymmdd}_b2_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b1_dict[s] = pd.read_csv(b1pred_k_path, index_col = 0)
                b2_dict[s] = pd.read_csv(b2pred_k_path, index_col = 0)
        # statspath = Path(outputDir) / f"{yyyymmdd}_coefCircCategorical_homolateral0_refCOMBINED_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
    elif len(predictors) == 3:
        beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b1pred_path = Path(outputDir) / f"{yyyymmdd}_b1_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b2pred_path = Path(outputDir) / f"{yyyymmdd}_b2_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        
        if len(slopes)>0:
            b1_dict = {};  b2_dict = {}
            for s in slopes:
                b1pred_k_path = Path(outputDir) / f"{yyyymmdd}_b1_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b2pred_k_path = Path(outputDir) / f"{yyyymmdd}_b2_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b1_dict[s] = pd.read_csv(b1pred_k_path, index_col = 0)
                b2_dict[s] = pd.read_csv(b2pred_k_path, index_col = 0)
                
    elif len(predictors) == 2:
        beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b1pred_path = Path(outputDir) / f"{yyyymmdd}_b1_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b2pred_path = Path(outputDir) / f"{yyyymmdd}_b2_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        
        if len(slopes)>0:
            b1_dict = {};  b2_dict = {}
            for s in slopes:
                b1pred_k_path = Path(outputDir) / f"{yyyymmdd}_b1_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b2pred_k_path = Path(outputDir) / f"{yyyymmdd}_b2_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b1_dict[s] = pd.read_csv(b1pred_k_path, index_col = 0)
                b2_dict[s] = pd.read_csv(b2pred_k_path, index_col = 0)
    
    else:
        raise ValueError("Invalid number of predictors supplied!")
    
    if np.any(['lSup' in s for s in predictors]):
        predictors = np.array(predictors)
        for i, s in enumerate(predictors):
            if 'lSup' in s:
                predictors[i] = 'limbSupportPC' + s[-1]
        predictors = list(predictors)
    
    beta1 = pd.read_csv(beta1path, index_col = 0)
    beta2 = pd.read_csv(beta2path, index_col = 0)
    if len(slopes)>0:
        b1pred = pd.read_csv(b1pred_path, index_col = 0)
        b2pred = pd.read_csv(b2pred_path, index_col = 0)
        
    if np.all(x_pred_range) == None:
        pred_x_num = 100 
    else:
        pred_x_num = np.max([len(x) for x in x_pred_range.values()])
    
    if ref == 'COMBINED' and categ_var=='refLimb':
        ref_iterables = ['', 'rH1']
    elif categ_var=='trialType':
        ref_iterables = ['', 'slope']
    else:
        ref_iterables = ['']
    
    # define the data storage arrays
    phase2_preds = np.empty((beta1.shape[0], 
                             pred_x_num,
                             len(predictors),
                             len(mice)+1,
                             len(ref_iterables)
                             ))
    phase2_preds[:] = np.nan
    
    x_range = np.empty((pred_x_num,
                             len(predictors),
                             ))
    x_range[:] = np.nan

    for b, predictor in enumerate(predictors):

        # define the predictor
        pred_id = np.where(np.asarray(predictors) == predictor)[0][0]
        pred = f'pred{pred_id+1}' 
        
        # define the range of snout-hump angles or inclines (x axis predictor)
        pred_relevant = utils_processing.remove_outliers(datafull[predictor]) if predictor != 'incline' else np.asarray(datafull[predictor])
        pred_centred = pred_relevant - np.nanmean(pred_relevant) #centering

        if np.all(x_pred_range) == None or predictor not in x_pred_range.keys():
            pred_range = np.linspace(pred_centred.min(), pred_centred.max(), num = pred_x_num)
        else:
            pred_range = x_pred_range[predictor] - np.nanmean(pred_relevant)

        x_range[:, b] = pred_range + np.nanmean(pred_relevant)
        
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
            nonpreds.append(f'pred{pred_id+1}') 
            nonpred_relevant = utils_processing.remove_outliers(datafull[other])
            nonpred_prct = np.percentile(nonpred_relevant, prcnt) - np.nanmean(nonpred_relevant)
            nonpred_dict[f'pred{pred_id+1}'] = nonpred_prct  
            # nonpred_max_dict[f'pred{pred_id+1}'] = nonpred_prct  
        
        for i, refLimb in enumerate(ref_iterables):       
            mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) +\
                  np.asarray(beta1[nonpreds[0]]).reshape(-1,1) * nonpred_dict[nonpreds[0]] +\
                  np.asarray(beta1[pred]).reshape(-1,1) @ pred_range.reshape(-1,1).T 

            mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) +\
                  np.asarray(beta2[nonpreds[0]]).reshape(-1,1) * nonpred_dict[nonpreds[0]] +\
                  np.asarray(beta2[pred]).reshape(-1,1) @ pred_range.reshape(-1,1).T 
       
            if len(predictors)>2:
               mu1 += np.asarray(beta1[nonpreds[1]]).reshape(-1,1) * nonpred_dict[nonpreds[1]] 
               mu2 += np.asarray(beta2[nonpreds[1]]).reshape(-1,1) * nonpred_dict[nonpreds[1]] 
           
            if len(predictors)>3:
               mu1 += np.asarray(beta1[nonpreds[2]]).reshape(-1,1) * nonpred_dict[nonpreds[2]] #+\
               mu2 += np.asarray(beta2[nonpreds[2]]).reshape(-1,1) * nonpred_dict[nonpreds[2]]    
              
            if len(predictors)>4:
               mu1 += np.asarray(beta1[nonpreds[3]]).reshape(-1,1) * nonpred_dict[nonpreds[3]] #+\
               mu2 += np.asarray(beta2[nonpreds[3]]).reshape(-1,1) * nonpred_dict[nonpreds[3]]  
       
            if interaction == 'TRUE' or  interaction == 'TRUEthreeway' or interaction == 'TRUEfourway':
                if pred in ['pred1', 'pred2']:
                    nonpred = np.setdiff1d(['pred1', 'pred2'], pred)[0]
                    mu1 += np.asarray(beta1["pred1:pred2"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])
                    mu2 += np.asarray(beta2["pred1:pred2"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])
                else:
                    mu1 += np.asarray(beta1["pred1:pred2"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2']
                    mu2 += np.asarray(beta2["pred1:pred2"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2']
                
            
            if interaction == 'TRUEsecondary' or interaction == 'TRUEthreeway' or interaction == 'TRUEfourway':
                if pred in ['pred2', 'pred3']:
                    nonpred = np.setdiff1d(['pred2', 'pred3'], pred)[0]
                    mu1 += np.asarray(beta1["pred2:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])
                    mu2 += np.asarray(beta2["pred2:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])
                else:
                    mu1 += np.asarray(beta1["pred2:pred3"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred3']
                    mu2 += np.asarray(beta2["pred2:pred3"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred3']
                
            if interaction == 'TRUEthreeway' or interaction == 'TRUEfourway':
                if pred in ['pred1', 'pred2', 'pred3']:
                    nonpredlist = np.setdiff1d(['pred1', 'pred2', 'pred3'], pred)
                    if pred in ['pred1', 'pred3']:
                        nonpred = np.setdiff1d(['pred1', 'pred3'], pred)[0]
                        mu1 += np.asarray(beta1["pred1:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])  
                        mu2 += np.asarray(beta2["pred1:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred]) 
                    else:
                        mu1 += np.asarray(beta1["pred1:pred3"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred3']
                        mu2 += np.asarray(beta2["pred1:pred3"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred3']
                    
                    mu1 += np.asarray(beta1["pred1:pred2:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                    mu2 += np.asarray(beta2["pred1:pred2:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                else:
                    mu1 += np.asarray(beta1["pred1:pred2:pred3"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2'] * nonpred_dict['pred3']
                    mu2 += np.asarray(beta2["pred1:pred2:pred3"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2'] * nonpred_dict['pred3']
            
            if interaction == 'TRUEfourway':
                if pred in ['pred1', 'pred2', 'pred4']:
                    nonpredlist = np.setdiff1d(['pred1', 'pred2', 'pred4'], pred)
                    if pred in ['pred1', 'pred4']:
                        nonpred = np.setdiff1d(['pred1', 'pred4'], pred)[0]
                        mu1 += np.asarray(beta1["pred1:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])  
                        mu2 += np.asarray(beta2["pred1:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred]) 
                    else:
                        mu1 += np.asarray(beta1["pred1:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred4']
                        mu2 += np.asarray(beta2["pred1:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred4']
                    
                    if pred in ['pred2', 'pred4']:
                        nonpred = np.setdiff1d(['pred2', 'pred4'], pred)[0]
                        mu1 += np.asarray(beta1["pred2:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])  
                        mu2 += np.asarray(beta2["pred2:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred]) 
                    else:
                        mu1 += np.asarray(beta1["pred2:pred4"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred4']
                        mu2 += np.asarray(beta2["pred2:pred4"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred4']
                    
                    mu1 += np.asarray(beta1["pred1:pred2:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                    mu2 += np.asarray(beta2["pred1:pred2:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                else:
                    mu1 += np.asarray(beta1["pred1:pred2:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2'] * nonpred_dict['pred4']
                    mu2 += np.asarray(beta2["pred1:pred2:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2'] * nonpred_dict['pred4']
            
                if pred in ['pred1', 'pred3', 'pred4']:
                    nonpredlist = np.setdiff1d(['pred1', 'pred3', 'pred4'], pred)
                    if pred in ['pred3', 'pred4']:
                        nonpred = np.setdiff1d(['pred3', 'pred4'], pred)[0]
                        mu1 += np.asarray(beta1["pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])  
                        mu2 += np.asarray(beta2["pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred]) 
                    else:
                        mu1 += np.asarray(beta1["pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred3'] * nonpred_dict['pred4']
                        mu2 += np.asarray(beta2["pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred3'] * nonpred_dict['pred4']
                    
                    mu1 += np.asarray(beta1["pred1:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                    mu2 += np.asarray(beta2["pred1:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                else:
                    mu1 += np.asarray(beta1["pred1:pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred3'] * nonpred_dict['pred4']
                    mu2 += np.asarray(beta2["pred1:pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred3'] * nonpred_dict['pred4']
            
                if pred in ['pred2', 'pred3', 'pred4']:
                    nonpredlist = np.setdiff1d(['pred2', 'pred3', 'pred4'], pred)
                    
                    mu1 += np.asarray(beta1["pred2:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                    mu2 += np.asarray(beta2["pred2:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                else:
                    mu1 += np.asarray(beta1["pred2:pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred3'] * nonpred_dict['pred4']
                    mu2 += np.asarray(beta2["pred2:pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred3'] * nonpred_dict['pred4']
            
                
                mu1 += np.asarray(beta1["pred1:pred2:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpreds[0]] * nonpred_dict[nonpreds[1]] * nonpred_dict[nonpreds[2]])
                mu2 += np.asarray(beta2["pred1:pred2:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpreds[0]] * nonpred_dict[nonpreds[1]] * nonpred_dict[nonpreds[2]])        
            
            if refLimb == ref_iterables[-1] and len(ref_iterables)>1:
                mu1 += np.asarray(beta1[f"pred{len(predictors)+1}{refLimb}"]).reshape(-1,1) #the mean of the third predictor (if present) is zero because it was centred before modelling
                mu2 += np.asarray(beta2[f"pred{len(predictors)+1}{refLimb}"]).reshape(-1,1) 
            
            if len(slopes)>0:
                phase2_preds[:,:, b, 0, i] = np.arctan2(mu2, mu1)
                for im, m in enumerate(mice):
                    # add random intercepts
                    mu1_mouse = mu1 + np.asarray(b1pred.iloc[im,:]).reshape(-1,1)# @ pred_range.reshape(-1,1).T
                    mu2_mouse = mu2 + np.asarray(b2pred.iloc[im,:]).reshape(-1,1) #@ pred_range.reshape(-1,1).T
                    
                    # find out which parameter to add the slope t
                    # is it the predictor?
                    if pred in slopes:
                        mu1_mouse += np.asarray(b1_dict[pred].iloc[im,:]).reshape(-1,1) @ pred_range.reshape(-1,1).T
                        mu2_mouse += np.asarray(b2_dict[pred].iloc[im,:]).reshape(-1,1) @ pred_range.reshape(-1,1).T
                    nonpred_slopes = np.intersect1d(slopes, list(nonpred_dict.keys()))
                    if len(nonpred_slopes)>0:
                        for s in nonpred_slopes:
                            mu1_mouse += np.asarray(b1_dict[s].iloc[im,:]).reshape(-1,1) * nonpred_dict[s]
                            mu2_mouse += np.asarray(b2_dict[s].iloc[im,:]).reshape(-1,1) * nonpred_dict[s]
                
                    phase2_preds[:,:, b, im+1, i] = np.arctan2(mu2_mouse, mu1_mouse)
            else:
                phase2_prediction = np.arctan2(mu2, mu1)
                phase2_preds[:,:, b, :, i] = np.repeat(phase2_prediction[:, :, np.newaxis], 
                                                    len(mice)+1,
                                                    axis = 2)

    return (x_range,
            phase2_preds)

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
    elif predictor == 'duty_ratio':
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
    
def get_circGLM_stats(
        predictors,
        yyyymmdd,
        limb,
        ref,
        samples = 11079,
        interaction = 'TRUE',
        appdx = '_egr3',
        datafrac = 0.6,
        categ_var = 'refLimb',
        slopes = ['pred2'],
        outputDir = Config.paths['passiveOpto_output_folder'],
        iterations = 1000,
        mice = Config.passiveOpto_config['egr3_mice'],
        sBA_split_str = 'sBAsplitFALSE'
        ):
    
    if len(slopes)>0:
        slope_appdx = f'SLOPE{"".join(slopes)}'
    else:
        slope_appdx = ''
    
    rfl_str = f"_{categ_var}" if categ_var != None else ""
    
    if len(predictors) == 5:
        coef_cat_path = Path(outputDir) / f"{yyyymmdd}_coefCircCategorical_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_{predictors[4]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        coef_cont_path = Path(outputDir) / f"{yyyymmdd}_coefCircContinuous_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_{predictors[4]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
    elif len(predictors) == 4:
        coef_cat_path = Path(outputDir) / f"{yyyymmdd}_coefCircCategorical_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        coef_cont_path = Path(outputDir) / f"{yyyymmdd}_coefCircContinuous_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
    elif len(predictors) == 3:
        coef_cat_path = Path(outputDir) / f"{yyyymmdd}_coefCircCategorical_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        coef_cont_path = Path(outputDir) / f"{yyyymmdd}_coefCircContinuous_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
    elif len(predictors) == 2:
        coef_cat_path = Path(outputDir) / f"{yyyymmdd}_coefCircCategorical_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        coef_cont_path = Path(outputDir) / f"{yyyymmdd}_coefCircContinuous_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
    else:
        raise ValueError("Invalid number of predictors supplied!")
    
    if categ_var != None:
        coef_cat = pd.read_csv(coef_cat_path, index_col =0)
    coef_cont = pd.read_csv(coef_cont_path, index_col =0)
    
    stat_dict = {}
    
    if categ_var != None:
        for coef in coef_cat.index:
            LB = round(coef_cat.loc[coef,'LB'],2)
            UB = round(coef_cat.loc[coef,'UB'],2)
            ptxt = "*" if np.sign(LB)==np.sign(UB) else "n.s."
            stat_dict[coef] = ptxt
        
    ssdolist = [x for x in coef_cont.index if "SSDO" in x]
    for ssdo in ssdolist:
        LB = round(coef_cont.loc[ssdo,'LB HPD'],2)
        UB = round(coef_cont.loc[ssdo,'UB HPD'],2)
        ptxt = "*" if np.sign(LB)==np.sign(UB) else "n.s."
        stat_dict[ssdo.split(" ")[0]] = ptxt
    return stat_dict

def get_circGLM_slopes_per_mouse(
        predictors,
        yyyymmdd,
        limb,
        ref,
        samples = 11079,
        interaction = 'TRUE',
        categ_var = "refLimb",
        appdx = '_egr3',
        datafrac = 0.6,
        slopes = ['pred2'],
        outputDir = Config.paths['passiveOpto_output_folder'],
        iterations = 1000,
        mice = Config.passiveOpto_config['egr3_mice'],
        x_pred_range = None,
        sBA_split_str = 'sBAsplitFALSE'
                ):
    """
    predictors (list of strs)
    yyyymmdd (str) : yyyy-mm-dd
    limb (str) : 'homolateral0', 'lF0', ...
    ref (str) : 'COMBINED', 'lH1', ...
    samples (int) : number of samples used to fit the model
    interaction (str) : 'TRUE' or 'FALSE'
    appdx (str) : '' or '_egr3' or 'egr3ctrl'
    datafrac (float) : fraction of data used to fit the model,
    slopes (list of strs) : 'pred1', 'pred2', 'pred1pred2' etc
    ----------------------------------
    IN THIS FUNCTION,
    min/max predictor range is computed for each predictor and for each mouse 
    separately, but all arrays have 100 vals, so arrays from the final multi-D 
    array cannot be aligned across mice!!
    good for plotting individual mouse data without averaging across mice!
    
    ADDED x_pred_range so that the min/max range can be modified as well! 
    if this is used, the func should be inside an iterator over mice 
    (this is quite inefficient, but a quick fix)
    ----------------------------------
    beta = fixed effects predictors
    b_pred = random effect intercept (assumed to always be present)
    """
    if len(slopes)>0:
        slope_appdx = f'SLOPE{"".join(slopes)}'
    else:
        slope_appdx = ''
    
    rfl_str = f"_{categ_var}" if categ_var != None else ""
    
    datafull = data_loader.load_processed_data(dataToLoad = 'strideParams',#'Merged',
                                               outputDir = outputDir,
                                               yyyymmdd = yyyymmdd,
                                               limb = ref, 
                                               appdx = appdx)[0]
    # datafull = pd.read_csv(r"C:\Users\MurrayLab\Documents\PassiveOptoTreadmill\2022-08-18_strideParams_incline_trialType_COMBINED.csv")
    # if 'deg' in datafull['headLVL'][0]:
    #     datafull['incline'] = [-int(x[3:]) for x in datafull['headLVL']]
    if 'headLVL' in datafull.columns:
        if 'deg' in datafull['headLVL'].iloc[-1] or 'deg' in datafull['headLVL'][0]:
            datafull['incline'] = [-int(x[3:]) for x in datafull['headLVL']]
    if 'trialType' in datafull.columns:
        if 'deg' in datafull['trialType'].iloc[-1] or 'deg' in datafull['trialType'][0]:
            datafull['incline'] = [-int(x[3:]) for x in datafull['trialType']]
    
    if len(predictors) == 4:
        beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b1pred_path = Path(outputDir) / f"{yyyymmdd}_b1_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b2pred_path = Path(outputDir) / f"{yyyymmdd}_b2_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        
        if len(slopes)>0:
            b1_dict = {};  b2_dict = {}
            for s in slopes:
                b1pred_k_path = Path(outputDir) / f"{yyyymmdd}_b1_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b2pred_k_path = Path(outputDir) / f"{yyyymmdd}_b2_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b1_dict[s] = pd.read_csv(b1pred_k_path, index_col = 0)
                b2_dict[s] = pd.read_csv(b2pred_k_path, index_col = 0)
        # statspath = Path(outputDir) / f"{yyyymmdd}_coefCircCategorical_homolateral0_refCOMBINED_{predictors[0]}_{predictors[1]}_{predictors[2]}_{predictors[3]}_refLimb_interaction{interaction}_continuous_randMouse_sBAsplitFALSE_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
    elif len(predictors) == 3:
        beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b1pred_path = Path(outputDir) / f"{yyyymmdd}_b1_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b2pred_path = Path(outputDir) / f"{yyyymmdd}_b2_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        
        if len(slopes)>0:
            b1_dict = {};  b2_dict = {}
            for s in slopes:
                b1pred_k_path = Path(outputDir) / f"{yyyymmdd}_b1_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b2pred_k_path = Path(outputDir) / f"{yyyymmdd}_b2_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}_{predictors[2]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b1_dict[s] = pd.read_csv(b1pred_k_path, index_col = 0)
                b2_dict[s] = pd.read_csv(b2pred_k_path, index_col = 0)
                
    elif len(predictors) == 2:
        beta1path = Path(outputDir) / f"{yyyymmdd}_beta1_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        beta2path = Path(outputDir) / f"{yyyymmdd}_beta2_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b1pred_path = Path(outputDir) / f"{yyyymmdd}_b1_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        b2pred_path = Path(outputDir) / f"{yyyymmdd}_b2_pred_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
        
        if len(slopes)>0:
            b1_dict = {};  b2_dict = {}
            for s in slopes:
                b1pred_k_path = Path(outputDir) / f"{yyyymmdd}_b1_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b2pred_k_path = Path(outputDir) / f"{yyyymmdd}_b2_{s}_{limb}_ref{ref}_{predictors[0]}_{predictors[1]}{rfl_str}_interaction{interaction}_continuous_randMouse{slope_appdx}_{sBA_split_str}_{datafrac}data{samples}s_{iterations}its_100burn_3lag.csv"
                b1_dict[s] = pd.read_csv(b1pred_k_path, index_col = 0)
                b2_dict[s] = pd.read_csv(b2pred_k_path, index_col = 0)
    
    else:
        raise ValueError("Invalid number of predictors supplied!")
    
    
    beta1 = pd.read_csv(beta1path, index_col = 0)
    beta2 = pd.read_csv(beta2path, index_col = 0)
    b1pred = pd.read_csv(b1pred_path, index_col = 0)
    b2pred = pd.read_csv(b2pred_path, index_col = 0)
    
    if np.all(x_pred_range) == None:
        pred_x_num = 100 
    else:
        pred_x_num = np.max([len(x) for x in x_pred_range.values()])
    
    if ref == 'COMBINED' and categ_var=='refLimb':
        ref_iterables = ['', 'rH1']
    elif categ_var=='trialType':
        ref_iterables = ['', 'slope']
    else:
        ref_iterables = ['']
    
    # define the data storage arrays
    phase2_preds = np.empty((beta1.shape[0], 
                             pred_x_num,
                             len(predictors),
                             len(mice)+1,
                             len(ref_iterables)
                             ))
    phase2_preds[:] = np.nan
    
    # define the predictor range storage arrays
    x_preds = np.empty((pred_x_num,
                        len(predictors),
                        len(mice)+1,
                        len(ref_iterables)
                        ))
    x_preds[:] = np.nan
    
    for b, predictor in enumerate(predictors):   
        # define the predictor
        pred_id = np.where(np.asarray(predictors) == predictor)[0][0]
        pred = f'pred{pred_id+1}' 
        
        # identify other predictors
        others = np.setdiff1d(predictors, predictor)
        nonpreds = []
        
        for im in range(len(mice)+1):
            if im == 0: # zeroth array should be the average over the entire range
                datafull_sub = datafull
            else:
                datafull_sub = datafull[datafull['mouseID'] == mice[im-1]]
            
            # define the range of snout-hump angles or inclines (x axis predictor)
            pred_relevant = utils_processing.remove_outliers(datafull_sub[predictor]) 
            pred_centred = pred_relevant - np.nanmean(pred_relevant) #centering
            if np.all(x_pred_range) == None or predictor not in x_pred_range.keys():
                pred_range = np.linspace(pred_centred.min(), pred_centred.max(), num = pred_x_num)
            else:
                pred_range = x_pred_range[predictor]
                
            # pred_range = np.linspace(pred_centred.min(), pred_centred.max(), num = pred_x_num)
            x_preds[:, b, im, :] = np.repeat(
                                    pred_range+np.nanmean(pred_relevant),
                                    len(ref_iterables)
                                    ).reshape(-1,len(ref_iterables))
            
            # find median param (excluding outliers) for the other predictors
            prcnt = 50
            nonpred_dict = {}
            nonpred_max_dict = {}
            for other in others:
                pred_id = np.where(np.asarray(predictors) == other)[0][0]
                nonpreds.append(f'pred{pred_id+1}') 
                nonpred_relevant = utils_processing.remove_outliers(datafull_sub[other])
                nonpred_prct = np.percentile(nonpred_relevant, prcnt) - np.nanmean(nonpred_relevant)
                nonpred_dict[f'pred{pred_id+1}'] = nonpred_prct  
                nonpred_max_dict[f'pred{pred_id+1}'] = nonpred_prct  
        
        # for i, refLimb in enumerate(ref_iterables):
        
            mu1 = np.asarray(beta1['(Intercept)']).reshape(-1,1) +\
                  np.asarray(beta1[nonpreds[0]]).reshape(-1,1) * nonpred_dict[nonpreds[0]] +\
                  np.asarray(beta1[pred]).reshape(-1,1) @ pred_range.reshape(-1,1).T 

            mu2 = np.asarray(beta2['(Intercept)']).reshape(-1,1) +\
                  np.asarray(beta2[nonpreds[0]]).reshape(-1,1) * nonpred_dict[nonpreds[0]] +\
                  np.asarray(beta2[pred]).reshape(-1,1) @ pred_range.reshape(-1,1).T 
       
            if len(predictors)>2:
               mu1 += np.asarray(beta1[nonpreds[1]]).reshape(-1,1) * nonpred_dict[nonpreds[1]] 
               mu2 += np.asarray(beta2[nonpreds[1]]).reshape(-1,1) * nonpred_dict[nonpreds[1]] 
           
            if len(predictors)>3:
               mu1 += np.asarray(beta1[nonpreds[2]]).reshape(-1,1) * nonpred_dict[nonpreds[2]] #+\
               mu2 += np.asarray(beta2[nonpreds[2]]).reshape(-1,1) * nonpred_dict[nonpreds[2]]     
       
            if interaction == 'TRUE' or  interaction == 'TRUEthreeway' or interaction == 'TRUEfourway':
                if pred in ['pred1', 'pred2']:
                    nonpred = np.setdiff1d(['pred1', 'pred2'], pred)[0]
                    mu1 += np.asarray(beta1["pred1:pred2"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])
                    mu2 += np.asarray(beta2["pred1:pred2"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])
                else:
                    mu1 += np.asarray(beta1["pred1:pred2"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2']
                    mu2 += np.asarray(beta2["pred1:pred2"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2']
                
            
            if interaction == 'TRUEsecondary' or interaction == 'TRUEthreeway' or interaction == 'TRUEfourway':
                if pred in ['pred2', 'pred3']:
                    nonpred = np.setdiff1d(['pred2', 'pred3'], pred)[0]
                    mu1 += np.asarray(beta1["pred2:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])
                    mu2 += np.asarray(beta2["pred2:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])
                else:
                    mu1 += np.asarray(beta1["pred2:pred3"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred3']
                    mu2 += np.asarray(beta2["pred2:pred3"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred3']
                
            if interaction == 'TRUEthreeway' or interaction == 'TRUEfourway':
                if pred in ['pred1', 'pred2', 'pred3']:
                    nonpredlist = np.setdiff1d(['pred1', 'pred2', 'pred3'], pred)
                    if pred in ['pred1', 'pred3']:
                        nonpred = np.setdiff1d(['pred1', 'pred3'], pred)[0]
                        mu1 += np.asarray(beta1["pred1:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])  
                        mu2 += np.asarray(beta2["pred1:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred]) 
                    else:
                        mu1 += np.asarray(beta1["pred1:pred3"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred3']
                        mu2 += np.asarray(beta2["pred1:pred3"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred3']
                    
                    mu1 += np.asarray(beta1["pred1:pred2:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                    mu2 += np.asarray(beta2["pred1:pred2:pred3"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                else:
                    mu1 += np.asarray(beta1["pred1:pred2:pred3"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2'] * nonpred_dict['pred3']
                    mu2 += np.asarray(beta2["pred1:pred2:pred3"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2'] * nonpred_dict['pred3']
            
            if interaction == 'TRUEfourway':
                if pred in ['pred1', 'pred2', 'pred4']:
                    nonpredlist = np.setdiff1d(['pred1', 'pred2', 'pred4'], pred)
                    if pred in ['pred1', 'pred4']:
                        nonpred = np.setdiff1d(['pred1', 'pred4'], pred)[0]
                        mu1 += np.asarray(beta1["pred1:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])  
                        mu2 += np.asarray(beta2["pred1:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred]) 
                    else:
                        mu1 += np.asarray(beta1["pred1:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred4']
                        mu2 += np.asarray(beta2["pred1:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred4']
                    
                    if pred in ['pred2', 'pred4']:
                        nonpred = np.setdiff1d(['pred2', 'pred4'], pred)[0]
                        mu1 += np.asarray(beta1["pred2:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])  
                        mu2 += np.asarray(beta2["pred2:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred]) 
                    else:
                        mu1 += np.asarray(beta1["pred2:pred4"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred4']
                        mu2 += np.asarray(beta2["pred2:pred4"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred4']
                    
                    mu1 += np.asarray(beta1["pred1:pred2:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                    mu2 += np.asarray(beta2["pred1:pred2:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                else:
                    mu1 += np.asarray(beta1["pred1:pred2:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2'] * nonpred_dict['pred4']
                    mu2 += np.asarray(beta2["pred1:pred2:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred2'] * nonpred_dict['pred4']
            
                if pred in ['pred1', 'pred3', 'pred4']:
                    nonpredlist = np.setdiff1d(['pred1', 'pred3', 'pred4'], pred)
                    if pred in ['pred3', 'pred4']:
                        nonpred = np.setdiff1d(['pred3', 'pred4'], pred)[0]
                        mu1 += np.asarray(beta1["pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred])  
                        mu2 += np.asarray(beta2["pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpred]) 
                    else:
                        mu1 += np.asarray(beta1["pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred3'] * nonpred_dict['pred4']
                        mu2 += np.asarray(beta2["pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred3'] * nonpred_dict['pred4']
                    
                    mu1 += np.asarray(beta1["pred1:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                    mu2 += np.asarray(beta2["pred1:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                else:
                    mu1 += np.asarray(beta1["pred1:pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred3'] * nonpred_dict['pred4']
                    mu2 += np.asarray(beta2["pred1:pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred1'] * nonpred_dict['pred3'] * nonpred_dict['pred4']
            
                if pred in ['pred2', 'pred3', 'pred4']:
                    nonpredlist = np.setdiff1d(['pred2', 'pred3', 'pred4'], pred)
                    
                    mu1 += np.asarray(beta1["pred2:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                    mu2 += np.asarray(beta2["pred2:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpredlist[0]] * nonpred_dict[nonpredlist[1]])
                else:
                    mu1 += np.asarray(beta1["pred2:pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred3'] * nonpred_dict['pred4']
                    mu2 += np.asarray(beta2["pred2:pred3:pred4"]).reshape(-1,1) * nonpred_dict['pred2'] * nonpred_dict['pred3'] * nonpred_dict['pred4']
            
                
                mu1 += np.asarray(beta1["pred1:pred2:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpreds[0]] * nonpred_dict[nonpreds[1]] * nonpred_dict[nonpreds[2]])
                mu2 += np.asarray(beta2["pred1:pred2:pred3:pred4"]).reshape(-1,1) @ (pred_range.reshape(-1,1).T * nonpred_dict[nonpreds[0]] * nonpred_dict[nonpreds[1]] * nonpred_dict[nonpreds[2]])        
            
            for i, refLimb in enumerate(ref_iterables):
                if refLimb == ref_iterables[-1] and len(ref_iterables)>1:
                    mu1 += np.asarray(beta1[f"pred{len(predictors)+1}{refLimb}"]).reshape(-1,1) #the mean of the third predictor (if present) is zero because it was centred before modelling
                    mu2 += np.asarray(beta2[f"pred{len(predictors)+1}{refLimb}"]).reshape(-1,1) 
                
                if im == 0:
                    phase2_preds[:,:, b, im, i] = np.arctan2(mu2, mu1)
                else:
                    mu1_mouse = mu1 + np.asarray(b1pred.iloc[im-1,:]).reshape(-1,1)# @ pred_range.reshape(-1,1).T
                    mu2_mouse = mu2 + np.asarray(b2pred.iloc[im-1,:]).reshape(-1,1) #@ pred_range.reshape(-1,1).T
                     
                    if len(slopes)>0:
                        # find out which parameter to add the slope t
                        # is it the predictor?
                        if pred in slopes:
                            mu1_mouse += np.asarray(b1_dict[pred].iloc[im-1,:]).reshape(-1,1) @ pred_range.reshape(-1,1).T
                            mu2_mouse += np.asarray(b2_dict[pred].iloc[im-1,:]).reshape(-1,1) @ pred_range.reshape(-1,1).T
                        nonpred_slopes = np.setdiff1d(slopes, np.concatenate(([pred],list(nonpred_dict.keys()))))
                        if len(nonpred_slopes)>0:
                            for s in nonpred_slopes:
                                mu1_mouse += np.asarray(b1_dict[s].iloc[im-1,:]).reshape(-1,1) * nonpred_dict[s]
                                mu2_mouse += np.asarray(b2_dict[s].iloc[im-1,:]).reshape(-1,1) * nonpred_dict[s]
                    
                    phase2_preds[:,:, b, im, i] = np.arctan2(mu2_mouse, mu1_mouse)


    return (x_preds,
            phase2_preds)
    