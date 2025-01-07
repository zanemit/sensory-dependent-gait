import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\thesis")

from processing import data_loader, utils_processing, forceplate_data_manager
from processing.data_config import Config

def get_angles_vs_param(outputDir=Config.paths["forceplate_output_folder"], 
                        param = 'headHW', 
                        yyyymmdd = None):
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
        if angles_df.columns.nlevels == 5:
            angles_df.columns = angles_df.columns.droplevel(0)
    else:
        raise ValueError("The supplied output directory does not exist!")
        #TODO need to check if weight calibration is required here!
    if param == 'headHW':
        headHW_df, _ = data_loader.load_processed_data(outputDir, 
                                                       dataToLoad='forceplateHeadHW', 
                                                       yyyymmdd=yyyymmdd)
        if headHW_df.columns.nlevels == 5:
             headHW_df.columns = headHW_df.columns.droplevel(0)
   
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
        
     
    df.to_csv(os.path.join(outputDir, yyyymmdd + f'_forceplateAngleParamsR_{param}.csv'))
        
def group_data_by_param(outputDir=Config.paths["forceplate_output_folder"], 
                        param = 'headHW', 
                        yyyymmdd = None):
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
        df, yyyymmdd = data_loader.load_processed_data(outputDir, 
                                                       dataToLoad='forceplateData', 
                                                       yyyymmdd = yyyymmdd)
        # import mouse weight metadata
        metadata_df = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                      dataToLoad = 'metadataProcessed', 
                                                      yyyymmdd = yyyymmdd)
        # weight calibration
        df_v, headplate_df = forceplate_data_manager.weight_calibrate_dataframe(
                                                        df, 
                                                        metadata_df,
                                                        yyyymmdd = yyyymmdd,
                                                        outputDir = Config.paths["forceplate_output_folder"]
                                                        )
    else:
        raise ValueError("The supplied output directory does not exist!")
    
    if df.columns.nlevels == 5:
         df.columns = df.columns.droplevel(0)
    if df_v.columns.nlevels == 5:
         df_v.columns = df_v.columns.droplevel(0)   
    
    mice = np.unique(df.columns.get_level_values(0))
    # if int(yyyymmdd[:4])>=2023:
    #     levels = np.unique(df.columns.get_level_values(2))
    # else:
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
        

    if param == 'snoutBodyAngle':
        angles_df, _ = data_loader.load_processed_data(outputDir, dataToLoad='forceplateAngles', yyyymmdd= yyyymmdd)
        if int(yyyymmdd[:4])>=2023:
            angles_df.columns = angles_df.columns.droplevel(0)
        paramCont_noOutliers = angles_noOutliers = utils_processing.remove_outliers(angles_df.mean())
        paramCont_num = 3
    elif param == 'headHW':
        if not 'rl' in levels[0]:
            raise ValueError("param == 'headHW' is allowed only for head height datasets!")     
        headHW_df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                       dataToLoad='forceplateHeadHW', 
                                                       yyyymmdd = yyyymmdd)
        if int(yyyymmdd[:4])>=2023:
            headHW_df.columns = headHW_df.columns.droplevel(0)
        paramCont_noOutliers = utils_processing.remove_outliers(headHW_df.mean())
        mice = np.unique(paramCont_noOutliers.index.get_level_values(0)) # repeated because some mice get lost here <_<
        paramCont_num = 3
    else:
        raise ValueError("Invalid param supplied!")
        
    # paramCont_split = np.linspace(paramCont_noOutliers.min(), paramCont_noOutliers.max(), paramCont_num+1)
    paramCont_split = [0.12844037, 0.44404985, 0.75965933, 1.07526882] # natural split for egr3
    # paramCont_split = [0.12844037, 0.37665906, 0.62487775, 0.87309645] # natural split for egr3...? (added Oct2024)
    # paramCont_split = [0.04255319, 0.41035078, 0.77814836, 1.14594595] # natural split for hhmice
    print(paramCont_split)
    
    slices = (slice(None), slice(None), slice(None))
    fore = df.loc[:, slices+('rF',)] + df.loc[:, slices+('lF',)].values
    hind = df.loc[:, slices+('rH',)] + df.loc[:, slices+('lH',)].values
    total = fore.values + hind.values
    fore_frac = fore / total
    hind_frac = np.ones_like(hind) - fore_frac
    total_df = pd.DataFrame(total, columns = fore.columns)
                                                                            
    fore_v = df_v.loc[:, slices+('rF',)] + df_v.loc[:, slices+('lF',)].values
    hind_v = df_v.loc[:, slices+('rH',)] + df_v.loc[:, slices+('lH',)].values

    # CoM computation
    x = df.loc[:, slices+('rF',)] + \
        df.loc[:, slices+('rH',)].values - \
        df.loc[:, slices+('lF',)].values - \
        df.loc[:, slices+('lH',)].values
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
        grouped_dict_fore = {}; grouped_dict_hind = {}
        grouped_dict_total = {}; grouped_dict_foreV = {}
        for key, val in group_col_ids.items():
            val = group_col_ids[key]
            if len(val)>0:
                grouped_dict_fore[key] = fore_frac[mouse].loc[:,val]
                grouped_dict_hind[key] = hind_frac[mouse].loc[:,val]
                grouped_dict_total[key] = total_df[mouse].loc[:,val]
                grouped_dict_foreV[key] = fore_v[mouse].loc[:,val]
            else:
                grouped_dict_fore[key] = pd.DataFrame(np.empty((fore_frac.shape[0],0)))
                grouped_dict_hind[key] = pd.DataFrame(np.empty((hind_frac.shape[0],0)))
                grouped_dict_total[key] = pd.DataFrame(np.empty((total_df.shape[0],0)))
                grouped_dict_foreV[key] = pd.DataFrame(np.empty((fore_v.shape[0],0)))
        # grouped_dict_fore = {key: fore_frac[mouse].loc[:, val] for key, val in group_col_ids.items()}
        # grouped_dict_hind = {key: hind_frac[mouse].loc[:, val] for key, val in group_col_ids.items()}
        # grouped_dict_total = {key: total_df[mouse].loc[:, val] for key, val in group_col_ids.items()}
        # grouped_dict_foreV = {key: fore_v[mouse].loc[:, val] for key, val in group_col_ids.items()}
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
  
def get_foot_placement_array(outputDir = Config.paths["forceplate_output_folder"], param=None):
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
    
    bp_level = 3 if int(yyyymmdd[:4])<2023 else 4
    bodyparts_in_df = np.unique(dfX.columns.get_level_values(bp_level))
    
    limbX = np.empty((dfX.shape[0]*int(dfX.shape[1]/len(bodyparts_in_df)), 4)) # 4 limb coordinates
    limbY = np.empty((dfY.shape[0]*int(dfY.shape[1]/len(bodyparts_in_df)), 4)) # 4 limb coordinates
    if int(yyyymmdd[:4])<2023:
        arrIDer = np.repeat(np.asarray(list(dfX.loc[:, (slice(None), slice(None), slice(None), 'rH1')].columns))[:,:3], dfX.shape[0], axis = 0)
        for i, limb in enumerate(limbs):
            limbX[:,i] = np.concatenate(np.vstack((np.asarray(dfX.loc[:, (slice(None), slice(None), slice(None), limb)]))).T)
            limbY[:,i] = np.concatenate(np.vstack((np.asarray(dfY.loc[:, (slice(None), slice(None), slice(None), limb)]))).T)
                
    else:
        arrIDer = np.repeat(np.asarray(list(dfX.loc[:, (slice(None), slice(None), slice(None), slice(None),'rH1')].columns))[:,:4], dfX.shape[0], axis = 0)
        for i, limb in enumerate(limbs):
            limbX[:,i] = np.concatenate(np.vstack((np.asarray(dfX.loc[:, (slice(None), slice(None), slice(None), slice(None), limb)]))).T)
            limbY[:,i] = np.concatenate(np.vstack((np.asarray(dfY.loc[:, (slice(None), slice(None), slice(None), slice(None), limb)]))).T)
        
        
    if param == 'snoutBodyAngle':
        bodyAngles, _ = data_loader.load_processed_data(outputDir, 
                                                        dataToLoad = 'forceplateAngles', 
                                                        yyyymmdd = yyyymmdd)
        multiindex_cols = bodyAngles.columns.droplevel(-1)
        
        if int(yyyymmdd[:4])<2023:
            arrIDer_df = pd.DataFrame(arrIDer, columns=['mouse', 'level', 'trial'])
            bodyAngles_col_df = pd.DataFrame(bodyAngles.columns.to_flat_index().map(lambda x: x[:-1]).tolist(),
                                             columns=['mouse', 'level', 'trial'])
        else:
            arrIDer_df = pd.DataFrame(arrIDer, columns=['date','mouse', 'level', 'trial'])
            bodyAngles_col_df = pd.DataFrame(bodyAngles.columns.to_flat_index().map(lambda x: x[:-1]).tolist(),
                                             columns=['date','mouse', 'level', 'trial'])
            
        SBA_means = bodyAngles.mean(axis=0)
        
        SBA_means = pd.DataFrame({
            'mean': SBA_means.values,
            'mouse': bodyAngles_col_df['mouse'],
            'level': bodyAngles_col_df['level'],
            'trial': bodyAngles_col_df['trial']
            })
        
        if int(yyyymmdd[:4])<2023:
            merged_DF = arrIDer_df.merge(SBA_means, on=['mouse', 'level', 'trial'], how='left')
            df_reg = pd.DataFrame(np.hstack((arrIDer, limbX, limbY)), columns = np.concatenate((['mouseID'],['level'],['trial'],[param], [x+'x' for x in limbs], [x+'y' for x in limbs])) ) 
        else:
            SBA_means['date'] = bodyAngles_col_df['date']
            merged_DF = arrIDer_df.merge(SBA_means, on=['date','mouse', 'level', 'trial'], how='left')
            df_reg = pd.DataFrame(np.hstack((arrIDer, limbX, limbY)), columns = np.concatenate((['expDate','mouseID'],['level'],['trial'],[param], [x+'x' for x in limbs], [x+'y' for x in limbs])) ) 
          
        arrIDer = merged_DF
        param = f"_{param}"
    else:
        df_reg = pd.DataFrame(np.hstack((arrIDer, limbX, limbY)), columns = np.concatenate((['mouseID'],['level'],['trial'], [x+'x' for x in limbs], [x+'y' for x in limbs])) )   
    df_reg.to_csv(os.path.join(outputDir, yyyymmdd + f'_limbPositionRegressionArray{param}.csv'))
    
def compute_mean_params(outputDir=Config.paths["forceplate_output_folder"], 
                        param='snoutBodyAngle', 
                        yyyymmdd = None,
                        trial_duration=5, 
                        bp = 'rF1') :
    """
    per-level plot of forelimbs vs hindlimbs, each with per-mouse-averages + across-mice-average
    param can be 'snoutBodyAngle', 'headHW' (if rl trials, not deg), or 'levels', OR 'posX' (bp position in the frame)

    PLOT TITLES ARE CHANGED SUCH THAT DEG-40 (upward incline) IS 40 DEG
    AND RL2 IS 0 mm, RL17 IS -15 mm, AND RL-13 IS 15 mm
    (nothing is changed in the original data arrays -- except in the new means_df dataframe)
    this is trickier in the regression plots - the changed values are used as numericals!
    """
    outputDir = Path(outputDir).resolve()
    if outputDir.exists():
        df, yyyymmdd = data_loader.load_processed_data(outputDir, 
                                                           dataToLoad='forceplateData',
                                                           yyyymmdd = yyyymmdd)
        # import mouse weight metadata
        metadata_df = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                      dataToLoad = 'metadataProcessed', 
                                                      yyyymmdd = yyyymmdd)
        # weight calibration
        df_v, headplate_df = forceplate_data_manager.weight_calibrate_dataframe(
                                                        df, 
                                                        metadata_df,
                                                        yyyymmdd = yyyymmdd,
                                                        outputDir = Config.paths["forceplate_output_folder"]
                                                        )
    else:
        raise ValueError("The supplied output directory does not exist!")
    
    if df.columns.nlevels == 5:
         df.columns = df.columns.droplevel(0)
    if df_v.columns.nlevels == 5:
         df_v.columns = df_v.columns.droplevel(0) 

    mice = np.unique(df.columns.get_level_values(0))
    levels = np.unique(df.columns.get_level_values(1))
    if 'rl' in levels[0]:
        lvl_str_drop = 2
        levels = np.sort([int(x[lvl_str_drop:]) for x in levels])[::-1]
        levels = ['rl' + str(x) for x in levels]

    elif 'deg' in levels[0]:
        lvl_str_drop = 3
        levels = np.sort([int(x[lvl_str_drop:]) for x in levels])[::-1]
        levels = ['deg' + str(x) for x in levels]

    # check that columns are matching before summing dataframes
    for r in range(3):
        if np.all(df.loc[:, (slice(None), slice(None), slice(None), 'rF')].columns.get_level_values(r) != df.loc[:, (slice(None), slice(None), slice(None), 'lF')].columns.get_level_values(r)) \
            or np.all(df.loc[:, (slice(None), slice(None), slice(None), 'rF')].columns.get_level_values(r) != df.loc[:, (slice(None), slice(None), slice(None), 'lH')].columns.get_level_values(r)) \
                or np.all(df.loc[:, (slice(None), slice(None), slice(None), 'rF')].columns.get_level_values(r) != df.loc[:, (slice(None), slice(None), slice(None), 'rH')].columns.get_level_values(r)):
            raise ValueError(
                "Will not be able to sum the dataframes! Columns do not match!")
    
    if param == 'levels':
        try:  # load everything related to snoutBodyAngles
            angles_df, _ = data_loader.load_processed_data(outputDir, 
                                                           dataToLoad='forceplateAngles', 
                                                           yyyymmdd=yyyymmdd)
            if int(yyyymmdd[:4])>=2023:
                angles_df.columns = angles_df.columns.droplevel(0)
            angles_noOutliers = utils_processing.remove_outliers(angles_df.mean())
        except:
            angles_df = None
    else:
        if param == 'snoutBodyAngle':
            angles_df, _ = data_loader.load_processed_data(
                                            outputDir, 
                                            dataToLoad='forceplateAngles', 
                                            yyyymmdd=yyyymmdd)
            if int(yyyymmdd[:4])>=2023:
                angles_df.columns = angles_df.columns.droplevel(0)
            paramCont_noOutliers = angles_noOutliers = utils_processing.remove_outliers(angles_df.mean())
        else:
            if param == 'headHW':
                if not 'rl' in levels[0]:
                    raise ValueError("param == 'headHW' is allowed only for head height datasets!")
                headHW_df, _ = data_loader.load_processed_data(outputDir, 
                                                               dataToLoad='forceplateHeadHW', 
                                                               yyyymmdd=yyyymmdd)
                if int(yyyymmdd[:4])>=2023:
                    headHW_df.columns = headHW_df.columns.droplevel(0)
                paramCont_noOutliers = utils_processing.remove_outliers(headHW_df.mean())
            elif param == 'posX':
                try:
                    dlc_posX, _ = data_loader.load_processed_data(outputDir, 
                                                                  dataToLoad = 'dlcPostureX', 
                                                                  yyyymmdd = yyyymmdd)
                    if int(yyyymmdd[:4])>=2023:
                        dlc_posX.columns = dlc_posX.columns.droplevel(0)
                except:
                    raise ValueError (f"Limb X position data not found at {outputDir}!")
                dlc_posX_sub = dlc_posX.loc[:, (slice(None), slice(None), slice(None), bp)]
                paramCont_noOutliers = utils_processing.remove_outliers(dlc_posX_sub.mean()) 
                
            try:  # load everything related to snoutBodyAngles
                angles_df, _ = data_loader.load_processed_data(outputDir, 
                                                               dataToLoad='forceplateAngles', 
                                                               yyyymmdd=yyyymmdd)
                if int(yyyymmdd[:4])>=2023:
                    angles_df.columns = angles_df.columns.droplevel(0)
                angles_noOutliers = angles_df.mean()
                # angles_noOutliers = utils_processing.remove_outliers(angles_df.mean())
            except:
                angles_df = None
                raise ValueError ('Angles not found! Problems could arise - this condition has not been tested!')
    
    # weight calibration
    fore = df.loc[:, (slice(None), slice(None), slice(None), 'rF')] +\
            df.loc[:, (slice(None), slice(None), slice(None), 'lF')].values
    hind = df.loc[:, (slice(None), slice(None), slice(None), 'rH')] +\
            df.loc[:, (slice(None), slice(None), slice(None), 'lH')].values
    total = fore.values + hind.values
    fore_frac = fore / total
    hind_frac = np.ones_like(hind) - fore_frac
    total_df = pd.DataFrame(total, columns = fore.columns)
    # smr_df_v, headplate_df = forceplate_data_manager.weight_calibrate_dataframe(smr_df, weightCalib, metadata_df)
    fore_v = df_v.loc[:, (slice(None), slice(None), slice(None), 'rF')] +\
            df_v.loc[:, (slice(None), slice(None), slice(None), 'lF')].values
    hind_v = df_v.loc[:, (slice(None), slice(None), slice(None), 'rH')] +\
            df_v.loc[:, (slice(None), slice(None), slice(None), 'lH')].values
    
    # CoM computation
    x = df.loc[:, (slice(None), slice(None), slice(None), 'rF')] + \
        df.loc[:, (slice(None), slice(None), slice(None), 'rH')].values - \
        df.loc[:, (slice(None), slice(None), slice(None), 'lF')].values - \
        df.loc[:, (slice(None), slice(None), slice(None), 'lH')].values
    y = fore - hind.values
    x = x/total  # normalise by the elementwise total
    y = y/total

    import warnings
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')

    # FOR LATER: regression analyses (figures D and D2)
    # dictionary where intervals are keys and values are smr means per mouse across trials
    point_means = []  # mean pressure over 5 s
    point_stds = []
    point_means_f = []  # mean pressure over 5 s
    point_stds_f = []
    point_means_t = [] # total pressure over 5 s
    point_stds_t = []
    point_means_fv = [] # forelimb pressure over 5 s
    point_stds_fv = []
    point_means_hv = [] # hindlimb pressure over 5 s
    point_stds_hv = []
    point_means_head = [] # hindlimb pressure over 5 s
    point_stds_head = []
    point_means_x = []  # mean CoM x pos over 5 s
    point_stds_x = []
    point_means_y = []  # mean CoM y pos over 5 s
    point_stds_y = []
    param_means = []  # level
    angle_means = []
    mouse_list = []  # mouse id
    level_list = []
    trial_list = []

    if param == 'levels':
        for lnum, lvl in enumerate(levels):
            print(f"Working on level {lvl}")

            if type(angles_df) == pd.core.frame.DataFrame:
                mouse_means_angles = np.empty((angles_df.shape[0], 0))

            for mnum, mouse in enumerate(mice):
                print(f"Working on mouse {mouse}")
                if type(angles_df) == pd.core.frame.DataFrame:
                    # PLOT ANGLES ACROSS HEAD LEVELS - ONE LINE PER MOUSE
                    # subset df for one mouse, 500 rows x all columns
                    angles_df_mouse = angles_df[mouse]
                    if lvl in angles_df_mouse.columns.get_level_values(0):
                        # convert multiindex into columns; 1 row x a subset of columns
                        angles_noOut_mouse = angles_noOutliers[mouse].reset_index()
                        index = pd.MultiIndex.from_arrays(np.asarray(angles_noOut_mouse.iloc[:, :3]).T, names=[
                                                          "level", "trial", "limb"])  # generate multilevel indices from columns
                        # create a multilevel dataframe - equivalent to the multiindex angles_noOut_mouse
                        angles_noOut_mouse_df = pd.DataFrame(angles_noOut_mouse.iloc[:, 3], columns=index)
                        angles_df_mouse = angles_df_mouse[angles_df_mouse.columns.intersection(angles_noOut_mouse_df.columns)]  # keep only the relevant columns; 500 rows x a subset of columns
                        mouse_means_angles = np.hstack((mouse_means_angles, np.nanmean(angles_df_mouse[lvl], axis=1).reshape(-1, 1)))

                if lvl in fore_frac[mouse].columns.get_level_values(0):
                    for trial in np.unique(fore_frac[mouse][lvl].columns.get_level_values(0)):
                        param_means.append(int(lvl[lvl_str_drop:]))
                        point_means.append(float(hind_frac[mouse][lvl][trial].mean()))
                        point_stds.append(float(hind_frac[mouse][lvl][trial].std()))
                        point_means_f.append(float(fore_frac[mouse][lvl][trial].mean()))
                        point_stds_f.append(float(fore_frac[mouse][lvl][trial].std()))
                        point_means_t.append(float(total_df[mouse][lvl][trial].mean()))
                        point_stds_t.append(float(total_df[mouse][lvl][trial].std()))
                        point_means_fv.append(float(fore_v[mouse][lvl][trial].mean()))
                        point_stds_fv.append(float(fore_v[mouse][lvl][trial].std()))
                        point_means_hv.append(float(hind_v[mouse][lvl][trial].mean()))
                        point_stds_hv.append(float(hind_v[mouse][lvl][trial].std()))
                        point_means_head.append(float(headplate_df[mouse][lvl][trial].mean()))
                        point_stds_head.append(float(headplate_df[mouse][lvl][trial].std()))
                        point_means_x.append(float(x[mouse][lvl][trial].mean()))
                        point_stds_x.append(float(x[mouse][lvl][trial].std()))
                        point_means_y.append(float(y[mouse][lvl][trial].mean()))
                        point_stds_y.append(float(y[mouse][lvl][trial].std()))
                        mouse_list.append(mouse)
                        level_list.append(lvl)
                        if type(angles_df) == pd.core.frame.DataFrame and lvl in angles_df_mouse.columns.get_level_values(0):
                            if trial in angles_df_mouse[lvl].columns.get_level_values(0):
                                angle_means.append(float(angles_df_mouse[lvl][trial].mean()))
                            else:
                                angle_means.append(np.nan)
                        else:
                            # just append the level - get perfect correlation later
                            angle_means.append(int(lvl[lvl_str_drop:]))

    if param == 'snoutBodyAngle' or param == 'headHW' or param == 'posX':
        for mnum, mouse in enumerate(mice):
            # PLOT SMR DATA ACROSS SNOUT-BODY-ANGLES (or headHW)
            # subset dataframe pertaining to one mouse
            paramCont_noOut_mouse = paramCont_noOutliers[mouse]

            # group smr data based on angles
          
            # the same as paramCont_xx for snoutBodyAngle, but different for headHW
            angles_noOut_mouse = angles_noOutliers[mouse]
            # angles_noOut_mouse.index = angles_noOut_mouse.index.set_levels(angles_noOut_mouse.index.levels[2].str.replace('snoutBodyAngle', 'rF'), level=2)  # replace level 2 labels to match smr data


            # get mean angle + mean smr frac + mouse name
            for tup in paramCont_noOut_mouse.index:
                # mean angle over 5 s
                param_means.append(paramCont_noOut_mouse[tup])
                try:
                    # same as param_means in the snoutBodyAngle condition!
                    angle_means.append(angles_noOut_mouse[(tup[0], tup[1], 'snoutBodyAngle')])
                except:
                    # same as param_means in the snoutBodyAngle condition!
                    angle_means.append(np.nan)
                point_means.append(np.asarray(hind_frac.loc[:, (mouse, tup[0], tup[1])].mean())[0])
                point_stds.append(np.asarray(hind_frac.loc[:, (mouse, tup[0], tup[1])].std())[0])
                point_means_f.append(np.asarray(fore_frac.loc[:, (mouse, tup[0], tup[1])].mean())[0])
                point_stds_f.append(np.asarray(fore_frac.loc[:, (mouse, tup[0], tup[1])].std())[0])
                point_means_t.append(np.asarray(total_df.loc[:, (mouse, tup[0], tup[1])].mean())[0])
                point_stds_t.append(np.asarray(total_df.loc[:, (mouse, tup[0], tup[1])].std())[0])
                point_means_fv.append(np.asarray(fore_v.loc[:, (mouse, tup[0], tup[1])].mean())[0])
                point_stds_fv.append(np.asarray(fore_v.loc[:, (mouse, tup[0], tup[1])].std())[0])
                point_means_hv.append(np.asarray(hind_v.loc[:, (mouse, tup[0], tup[1])].mean())[0])
                point_stds_hv.append(np.asarray(hind_v.loc[:, (mouse, tup[0], tup[1])].std())[0])
                point_means_head.append(np.asarray(headplate_df.loc[:, (mouse, tup[0], tup[1])].mean())[0])
                point_stds_head.append(np.asarray(headplate_df.loc[:, (mouse, tup[0], tup[1])].std())[0])
                point_means_x.append(np.asarray(x.loc[:, (mouse, tup[0], tup[1])].mean())[0])
                point_stds_x.append(np.asarray(x.loc[:, (mouse, tup[0], tup[1])].std())[0])
                point_means_y.append(np.asarray(y.loc[:, (mouse, tup[0], tup[1])].mean())[0])
                point_stds_y.append(np.asarray(y.loc[:, (mouse, tup[0], tup[1])].std())[0])
                mouse_list.append(mouse)
                level_list.append(tup[0])
                trial_list.append(tup[1])

    # PROCESS THE MOUSE-ANGLE-FORCE DF
    if 'rl' in levels[0] and param == 'levels':
        param_means = -np.asarray(param_means)+2
    elif 'deg' in levels[0] and param == 'levels':
        param_means = -np.asarray(param_means)
        
    if param != 'posX':
        bp = ""
    df_means = pd.DataFrame(zip(mouse_list, 
                                level_list, 
                                param_means, 
                                angle_means, 
                                point_means, 
                                point_stds, 
                                point_means_f, 
                                point_stds_f, 
                                point_means_t, 
                                point_stds_t, 
                                point_means_fv, 
                                point_stds_fv, 
                                point_means_hv, 
                                point_stds_hv, 
                                point_means_head, 
                                point_stds_head, 
                                point_means_x, 
                                point_stds_x, 
                                point_means_y, 
                                point_stds_y),
                            columns=['mouse', 
                                     'level', 
                                     'param', 
                                     'angle', 
                                     'hind_frac', 
                                     'hind_std', 
                                     'fore_frac', 
                                     'fore_std', 
                                     'total_pressure', 
                                     'total_pressure_std', 
                                     'fore_weight_frac', 
                                     'fore_weight_std', 
                                     'hind_weight_frac', 
                                     'hind_weight_std', 
                                     'headplate_weight_frac', 
                                     'headplate_weight_std', 
                                     'CoMx_mean', 
                                     'CoMx_std', 
                                     'CoMy_mean', 
                                     'CoMy_std'])
    df_means.to_csv(os.path.join(outputDir, f"{yyyymmdd}_meanParamDF_{param}{bp}.csv"))

def get_dlc_posture_data(directory = r"F:\Forceplate\batch_egr3", 
                         yyyymmdd = None, 
                         output_directory = r"C:\Users\MurrayLab\Documents\Forceplate"):
    if type(directory) == str:
        dlc_files = [f for f in os.listdir(directory) if "video" in f and f.endswith('.h5')]
    else:
        raise ValueError("DLC file directory not supplied!")

    # load sample data to get fps
    dlc_tuples = []
    post_tuples = []
    trial_duration = 5
    fps = Config.forceplate_config["fps"]
    trig_num = trial_duration * fps
    bps_posture_polygon = ['b2','b3','body','b4','b5','b6','b7','b8','tailbase','rH2','rH1','rF2','rF1','snout']  
    bps_posts = ['postF1', 'postF2', 'postH1', 'postH2']
    
    dlc_posture_polygon_X = np.empty((trig_num, len(dlc_files)*len(bps_posture_polygon)))
    dlc_posture_polygon_Y = np.empty((trig_num, len(dlc_files)*len(bps_posture_polygon)))
    dlc_posts = np.empty((trig_num, len(dlc_files)*len(bps_posts)*2))
    dlc_column = 0; post_column = 0
    for f in dlc_files:
        if 'deg' in f:
            print(f"Grabbing data from file: {f[:18]}...")
        elif 'rl' in f:
            print(f"Grabbing data from file: {f[:24]}...")
        path = os.path.join(directory, f)
        dlc, _ = utils_processing.preprocess_dlc_data(path)
        
        # metadata
        fsplit = f.split('_')
        if int(yyyymmdd[:4])>= 2023:
            d_exp = fsplit[0]
            mouse_id = fsplit[1]
            level = fsplit[2]
            if 'deg' in f:
                trial = fsplit[3]
            elif 'rl' in f:
                trial = fsplit[3]
        else:
            mouse_id = fsplit[0]
            level = fsplit[1]
            if 'deg' in f:
                trial = fsplit[2]
            elif 'rl' in f:
                trial = fsplit[3]
        
        for bp in bps_posture_polygon:
            if int(yyyymmdd[:4])>= 2023:
                dlc_tuples.append((d_exp, mouse_id, level, trial, bp))
            else:
                dlc_tuples.append((mouse_id, level, trial, bp))
            dlc_posture_polygon_X[:,dlc_column] = dlc[bp]['x'][:trig_num]
            dlc_posture_polygon_Y[:,dlc_column] = -dlc[bp]['y'][:trig_num] +600
            dlc_column += 1
        
        for bp in bps_posts:
            if int(yyyymmdd[:4])>= 2023:
                post_tuples.append((d_exp, mouse_id, level, trial, bp, 'x'))
                post_tuples.append((d_exp, mouse_id, level, trial, bp, 'y'))
            else:
                post_tuples.append((mouse_id, level, trial, bp, 'x'))
                post_tuples.append((mouse_id, level, trial, bp, 'y'))
            dlc_posts[:, post_column] = dlc[bp]['x'][:trig_num]
            dlc_posts[:, post_column+1] = -dlc[bp]['y'][:trig_num] +600
            post_column += 2
    
    if yyyymmdd == None:
        yyyymmdd = input(['2023-11-06', '2022-04-02', '2022-04-04', '2021-10-26']) #rl12 on 2nd Apr, rl5 on 4th Apr
    
    if int(yyyymmdd[:4])>= 2023:
        dlc_index = pd.MultiIndex.from_tuples(
                                dlc_tuples, 
                                names=["expDate", "mouse", "level", "trial", "bodypart"]
                                )
        post_index = pd.MultiIndex.from_tuples(
                                post_tuples, 
                                names=["expDate", "mouse", "level", "trial", "bodypart", "coord"]
                                )
        
    else:
        dlc_index = pd.MultiIndex.from_tuples(
                                dlc_tuples, 
                                names=["mouse", "level", "trial", "bodypart"]
                                )
        post_index = pd.MultiIndex.from_tuples(
                                post_tuples, 
                                names=["mouse", "level", "trial", "bodypart", "coord"]
                                )
    dlcX_df = pd.DataFrame(dlc_posture_polygon_X, columns=dlc_index)
    dlcX_df.to_csv(os.path.join(output_directory, yyyymmdd + '_dlcPostureX.csv'))  
    dlcY_df = pd.DataFrame(dlc_posture_polygon_Y, columns=dlc_index)  
    dlcY_df.to_csv(os.path.join(output_directory, yyyymmdd + '_dlcPostureY.csv')) 
    
    post_df = pd.DataFrame(dlc_posts, columns=post_index)
    post_df.to_csv(os.path.join(output_directory, yyyymmdd + '_dlcPosts.csv'))

