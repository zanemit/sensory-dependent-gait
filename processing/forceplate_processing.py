import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from processing import data_loader, utils_processing,forceplate_data_manager
from processing.data_config import Config

def get_angles_vs_param(outputDir=Config.paths["forceplate_output_folder"], param = 'headHW', yyyymmdd = None):
    """
    reorganises the weight-adjusted head height/head levels and body tilt angle dataframes
    
    PARAMETERS:
        outputDir (str) : folder for newly generated files
        param (str) : 'headHW' or 'headLVL'
        yyyymmdd (str) : date of the data acquisition
                                            
    WRITES:
        "{yyyymmdd}_forceplateAngleParamsR.csv" : parameters to reconstruct the fitted calibration equation
    """
    outputDir = Path(outputDir).resolve()
    if outputDir.exists():
        angles_df, yyyymmdd = data_loader.load_processed_data(outputDir, dataToLoad='forceplateAngles')
    else:
        raise ValueError("The supplied output directory does not exist!")
        
    if param == 'headHW':
        headHW_df, _ = data_loader.load_processed_data(outputDir, dataToLoad='forceplateHeadHW', yyyymmdd=yyyymmdd)
        if 'rl' not in angles_df.columns.get_level_values(1)[0]:
            raise ValueError("param = 'headHW' requires a head height dataset!")

    if 'rl' in angles_df.columns.get_level_values(1)[0]:
        rename_dict = {lvl : -int(lvl[2:])+2 for lvl in np.unique(angles_df.columns.get_level_values(1))}
    elif 'deg' in angles_df.columns.get_level_values(1)[0]:
        rename_dict = {lvl : -int(lvl[3:]) for lvl in np.unique(angles_df.columns.get_level_values(1))}
    else:
        raise ValueError("Some strange data supplied!")
            
    mice = np.unique(angles_df.columns.get_level_values(0))
    angles_df = angles_df.rename(columns = rename_dict)
    
    headHWs_all = np.asarray([0,0])
    if param == 'headHW':
        headHW_df = headHW_df.rename(columns = rename_dict)
        headHWs_all = np.empty(())
    angles_all = np.empty(())
    lvl_all = np.empty(())
    mouseIDs = np.empty(())
    for m in mice:
        angle_means = []
        headHW_means = []
        lvl_means = []
        angles_df_sub = angles_df.loc[:, m]
        levels = np.unique(angles_df_sub.columns.get_level_values(0))
        for lvl in levels:
            angles = np.mean(angles_df_sub.loc[:, lvl], axis=0)
            angles_all = np.append(angles_all, angles)
            angle_means.append(angles.mean())

            if param == 'headHW':
                headHW = np.mean(headHW_df.loc[:, (m, lvl)], axis=1)
                headHW_means.append(headHW)
                headHWs_all = np.append(headHWs_all, np.asarray(headHW_df.loc[:, (m, lvl)]))

            lvl_means.append(lvl)
            lvl_all = np.append(lvl_all, np.repeat(lvl, len(angles)))
            mouseIDs = np.append(mouseIDs, np.repeat(m, len(angles)))
    
    df = pd.DataFrame(zip(mouseIDs[1:], angles_all[1:], lvl_all[1:]), columns=['mouseID', 'snoutBodyAngle', 'headLVL'])

    if param == 'headHW':
        df = pd.DataFrame(zip(mouseIDs[1:], angles_all[1:], headHWs_all[1:], lvl_all[1:]), columns=['mouseID', 'snoutBodyAngle', 'headHW', 'headLVL'])
    else:   
        df.to_csv(os.path.join(outputDir, yyyymmdd + f'_forceplateAngleParamsR_{param}.csv'))
        
def group_data_by_param(outputDir=Config.paths["forceplate_output_folder"], param = 'headHW', yyyymmdd = None):
    """
    reorganises the weight-adjusted head height/head levels and body tilt angle dataframes
    
    PARAMETERS:
        outputDir (str) : folder for newly generated files
        param (str) : 'headHW' or 'snoutBodyAngle' 
        yyyymmdd (str) : date of the data acquisition
                                            
    WRITES:
        "{yyyymmdd}_forceplateData_{param}.csv" : parameters to reconstruct the fitted calibration equation
    """
    outputDir = Path(outputDir).resolve()
    if outputDir.exists():
        df, yyyymmdd = data_loader.load_processed_data(outputDir, dataToLoad='forceplateData', yyyymmdd = yyyymmdd)
    else:
        raise ValueError("The supplied output directory does not exist!")

    mice = np.unique(df.columns.get_level_values(0))
    levels = np.unique(df.columns.get_level_values(1))

    # make level values more interpretable
    if 'rl' in levels[0]:
        lvl_str_drop = 2
        levels = np.sort([int(x[lvl_str_drop:]) for x in levels])[::-1]
        levels = ['rl' + str(x) for x in levels]

    elif 'deg' in levels[0]:
        lvl_str_drop = 3
        levels = np.sort([int(x[lvl_str_drop:]) for x in levels])[::-1]
        levels = ['deg' + str(x) for x in levels]

    # import weight-voltage calibration files
    weightCalib, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                     dataToLoad = 'weightCalibration', 
                                                     yyyymmdd = yyyymmdd)
    # import mouse weight metadata
    metadata_df = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                  dataToLoad = 'metadataProcessed', 
                                                  yyyymmdd = yyyymmdd)

    if param == 'snoutBodyAngle':
        angles_df, _ = data_loader.load_processed_data(outputDir, dataToLoad='forceplateAngles', yyyymmdd= yyyymmdd)
        paramCont_noOutliers = angles_noOutliers = utils_processing.remove_outliers(angles_df.mean())
        paramCont_num = 5
    elif param == 'headHW':
        if not 'rl' in levels[0]:
            raise ValueError("param == 'headHW' is allowed only for head height datasets!")     
        headHW_df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                       dataToLoad='forceplateHeadHW', 
                                                       yyyymmdd = yyyymmdd)
        paramCont_noOutliers = utils_processing.remove_outliers(headHW_df.mean())
        paramCont_num = 4
    else:
        raise ValueError("Invalid param supplied!")
        
    paramCont_split = np.linspace(paramCont_noOutliers.min(), paramCont_noOutliers.max(), paramCont_num+1)

    fore = df.loc[:, (slice(None), slice(None), slice(None), 'rF')] + df.loc[:, (slice(None), slice(None), slice(None), 'lF')].values
    hind = df.loc[:, (slice(None), slice(None), slice(None), 'rH')] + df.loc[:, (slice(None), slice(None), slice(None), 'lH')].values
    total = fore.values + hind.values
    fore_frac = fore / total
    hind_frac = np.ones_like(hind) - fore_frac
    total_df = pd.DataFrame(total, columns = fore.columns)
    # weight calibration
    df_v, headplate_df = forceplate_data_manager.weight_calibrate_dataframe(df, weightCalib, metadata_df)
    fore_v = df_v.loc[:, (slice(None), slice(None), slice(None), 'rF')] + df_v.loc[:, (slice(None), slice(None), slice(None), 'lF')].values
    hind_v = df_v.loc[:, (slice(None), slice(None), slice(None), 'rH')] + df_v.loc[:, (slice(None), slice(None), slice(None), 'lH')].values

    # CoM computation
    x = df.loc[:, (slice(None), slice(None), slice(None), 'rF')] + \
        df.loc[:, (slice(None), slice(None), slice(None), 'rH')].values - \
        df.loc[:, (slice(None), slice(None), slice(None), 'lF')].values - \
        df.loc[:, (slice(None), slice(None), slice(None), 'lH')].values
    y = fore - hind.values
    x = x/total  # normalise by the elementwise total
    y = y/total

    # FOR LATER: mean + CI plots
    tuples = []; tuples_CoM = []
    data = np.empty((df.shape[0], mice.shape[0] * len(levels) * 8))
    data_CoM = np.empty((df.shape[0], int(df.shape[1]/2)))
    import warnings
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')

    i = 0  ; k = 0     
    for mnum, mouse in enumerate(mice):
        # PLOT SMR DATA ACROSS headHW
        # subset dataframe pertaining to one mouse
        paramCont_noOut_mouse = paramCont_noOutliers[mouse]
        paramCont_noOut_mouse.index = paramCont_noOut_mouse.index.set_levels(
            paramCont_noOut_mouse.index.levels[2].str.replace(param, 'rF'), level=2)  # replace level 2 labels to match smr data
        paramCont_grouped = paramCont_noOut_mouse.groupby(pd.cut(paramCont_noOut_mouse, paramCont_split))  # group data based on angles
        group_col_ids = paramCont_grouped.groups  # same
        # group smr data based on angles
        grouped_dict_fore = {key: fore_frac[mouse].loc[:, val] for key, val in group_col_ids.items()}
        grouped_dict_hind = {key: hind_frac[mouse].loc[:, val] for key, val in group_col_ids.items()}
        grouped_dict_total = {key: total_df[mouse].loc[:, val] for key, val in group_col_ids.items()}
        grouped_dict_foreV = {key: fore_v[mouse].loc[:, val] for key, val in group_col_ids.items()}
        hind_v.columns = pd.MultiIndex.from_tuples([(x,y,z,'rF') for x,y,z,e in hind.columns], names = hind.columns.names) # change 'rH' to 'rF'
        headplate_df.columns = pd.MultiIndex.from_tuples([(x,y,z,'rF') for x,y,z,e in headplate_df.columns], names = headplate_df.columns.names) # change 'headplate' to 'rF'
        grouped_dict_hindV = {key: hind_v[mouse].loc[:, val] for key, val in group_col_ids.items()}
        grouped_dict_head = {key: headplate_df[mouse].loc[:, val] for key, val in group_col_ids.items()}

        # the same as paramCont_xx for snoutBodyAngle, but different for headHW
        # angles_noOut_mouse = angles_noOutliers[mouse]
        # angles_noOut_mouse.index = angles_noOut_mouse.index.set_levels(angles_noOut_mouse.index.levels[2].str.replace('snoutBodyAngle', 'rF'), level=2)  # replace level 2 labels to match smr data

        # # FOR LATER: assign intervals to a variable and generate neat titles
        conditions = list(grouped_dict_fore.keys())
        cond_titles = [f'{interval.left:.2f}-{interval.right:.2f}' for interval in conditions]
        
        # CoM
        grouped_dict_x = {key: x[mouse].loc[:, val] for key, val in group_col_ids.items()}
        grouped_dict_y = {key: y[mouse].loc[:, val] for key, val in group_col_ids.items()}

        for inum, interval in enumerate(grouped_dict_fore.keys()):
            tuples.append((interval, mouse, 'forelimbs'))
            tuples.append((interval, mouse, 'hindlimbs'))
            tuples.append((interval, mouse, 'total'))
            tuples.append((interval, mouse, 'fore_weight_frac'))
            tuples.append((interval, mouse, 'hind_weight_frac'))
            tuples.append((interval, mouse, 'headplate_weight_frac'))
            tuples.append((interval, mouse, 'CoMx'))
            tuples.append((interval, mouse, 'CoMy'))
            data[:, i*8] = np.nanmean(grouped_dict_fore[interval], axis=1)
            data[:, (i*8)+1] = np.nanmean(grouped_dict_hind[interval], axis=1)
            data[:, (i*8)+2] = np.nanmean(grouped_dict_total[interval], axis=1)
            data[:, (i*8)+3] = np.nanmean(grouped_dict_foreV[interval], axis=1)
            data[:, (i*8)+4] = np.nanmean(grouped_dict_hindV[interval], axis=1)
            data[:, (i*8)+5] = np.nanmean(grouped_dict_head[interval], axis=1)
            data[:, (i*8)+6] = np.nanmean(grouped_dict_x[interval], axis=1)
            data[:, (i*8)+7] = np.nanmean(grouped_dict_y[interval], axis=1)
            i +=1
            
            addcols = grouped_dict_x[interval].shape[1]
            [tuples_CoM.append((interval, mouse, ac, 'CoMx')) for ac in range(addcols)]
            [tuples_CoM.append((interval, mouse, ac, 'CoMy')) for ac in range(addcols)]
            data_CoM[:, k:(k+addcols)] = grouped_dict_x[interval].values
            data_CoM[:, (k+addcols):(k+addcols+addcols)] = grouped_dict_y[interval].values
            k += addcols*2

    # PROCESS THE DATA AND STDS DFs
    # get rid of redundant columns (col num is initialised as max possible but not all mice do all head levels)
    data = data[:, :len(tuples)]
    index = pd.MultiIndex.from_tuples(tuples, names=["level", "mouse", "limb"])
    # one smr data column per mouse per level/interval
    data = pd.DataFrame(data, columns=index) 
    data.to_csv(os.path.join(outputDir, f"{yyyymmdd}_forceplateData_groupedby_{param}.csv"))
    
    data_CoM = data_CoM[:, :len(tuples_CoM)]
    index = pd.MultiIndex.from_tuples(tuples_CoM, names=["level", "mouse", "trial","limb"])
    data_CoM = pd.DataFrame(data_CoM, columns=index) 
    data_CoM.to_csv(os.path.join(outputDir, f"{yyyymmdd}_forceplateData_CoMgrouped_{param}.csv"))
  
def get_foot_placement_array(outputDir = Config.paths["forceplate_output_folder"]):
    """
    stacks limb xy position data during locomotion or before optostim
    
    PARAMETERS
    ----------
    outputDir (str, optional) : path to output folder
    appdx (str) : "" (empty) or "incline" for head height and incline trials respectively
    
    WRITES
    ----------
    f'{yyyymmdd}_limbPositionRegressionArray_{data}_{param}{appdx}.csv'
    """    
    limbs = ['rH1', 'rH2', 'rF1', 'rF2'] # limbX and limbY have this order too
    
    # LOAD THE DATA
    dfX, yyyymmdd = data_loader.load_processed_data(outputDir, dataToLoad = 'dlcPostureX')
    dfY, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'dlcPostureY', yyyymmdd = yyyymmdd)
    bodyparts_in_df = np.unique(dfX.columns.get_level_values(3))
    
    limbX = np.empty((dfX.shape[0]*int(dfX.shape[1]/len(bodyparts_in_df)), 4)) # 4 limb coordinates
    limbY = np.empty((dfY.shape[0]*int(dfY.shape[1]/len(bodyparts_in_df)), 4)) # 4 limb coordinates
    arrIDer = np.repeat(np.asarray(list(dfX.loc[:, (slice(None), slice(None), slice(None), 'rH1')].columns))[:,:3], dfX.shape[0], axis = 0)
    for i, limb in enumerate(limbs):
        limbX[:,i] = np.concatenate(np.vstack((np.asarray(dfX.loc[:, (slice(None), slice(None), slice(None), limb)]))).T)
        limbY[:,i] = np.concatenate(np.vstack((np.asarray(dfY.loc[:, (slice(None), slice(None), slice(None), limb)]))).T)
    
    df_reg = pd.DataFrame(np.hstack((arrIDer, limbX, limbY)), columns = np.concatenate((['mouseID'],['level'],['trial'], [x+'x' for x in limbs], [x+'y' for x in limbs])) )   
    df_reg.to_csv(os.path.join(outputDir, yyyymmdd + '_limbPositionRegressionArray.csv'))
    