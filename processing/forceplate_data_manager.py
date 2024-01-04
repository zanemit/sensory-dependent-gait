import neo
import os
import pandas as pd
import numpy as np
from pathlib import Path
from processing.data_config import Config
from processing import data_loader, data_quality_checkers, process_metadata, utils_processing, utils_math


def get_data(dataDir, outputDir = Config.paths["forceplate_data_folder"], yyyymmdd = None):
    """
    loads data from all .csv files in the given directory
    organises it in a 4-level dataframe (mouse-headlevel-trial-limb)

    PARAMETERS:
        dataDir (str) : path to the data folder
        outputDir (bool or str) : path to .h5 file folder
        yyyymmdd : 
    RETURNS:
        df (multilevel dataframe) : as described above
        index (multiIndex object) : a list of all indices across levels
        fps (float) : sampling rate
    """
    if os.path.isdir(dataDir):
        csv_files = [f for f in os.listdir(dataDir) if f.endswith('.csv')]
        dlc_files = [f for f in os.listdir(dataDir) if f.endswith('.h5')]
    else:
        raise ValueError("Invalid directory entered!")

    # mouse weights reorganised with the passiveOpto code and moved to the forceplate dir
    metadata_df = data_loader.load_processed_data(outputDir, 
                                                  dataToLoad='metadataProcessed',
                                                  yyyymmdd = yyyymmdd)
    fps_db = pd.read_hdf(os.path.join(dataDir, 'fps_database.h5'))
    
    # import weight calibration data (weight-signal relationship)
    # weightCalib, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
    #                                                  dataToLoad = 'weightCalibration', 
    #                                                  yyyymmdd = yyyymmdd)

    # check if all files have the same frame rate
    if not np.all(np.diff(fps_db["fps"].astype(float)) == 0):
        print(f'Spike2 sampling rate is variable! Found these values: {np.unique(fps_db["fps"].astype(float))} !')
    
    sample_num = int(float(fps_db["fps"].min())*Config.forceplate_setup["trial_duration"]) # number of samples in 5 seconds
    print(f'Using a {sample_num/Config.forceplate_setup["trial_duration"]} Hz frame rate')

    dlc_tuples = []
    dlc_data = np.empty((Config.forceplate_config['fps']*Config.forceplate_setup["trial_duration"], len(dlc_files)))

    csv_tuples = []
    headHW_tuples = []
    csv_data = np.empty((sample_num, len(csv_files)*4))
    headHW_data = np.empty((1, len(dlc_files)))

    failed_count = 0
    csv_column = 0
    dlc_column = 0
    for f in csv_files:
        print("Grabbing data from file: ", f[:-4])
        path = os.path.join(dataDir, f)
        signals = pd.read_csv(path)
        fps = fps_db[fps_db['filename'] == f[:-3] + 'smr']['fps'][0]
        if type(fps) != str:
            fps = float(fps.iloc[-1])
        else:
            fps = float(fps)

        # metadata
        fsplit = f.split('_')
        mouse_id = fsplit[0]
        level = fsplit[1]
        trial = fsplit[-1][0]

        start_sample = 0

        # process videos
        f_dlc = [d for d in dlc_files if f[:-4] in d][0]
        path_dlc = os.path.join(dataDir, f_dlc)
        dlc, _ = utils_processing.preprocess_dlc_data(path_dlc)

        if 'camTrig' in signals.columns:
            rising_edge_ids, falling_edge_ids, _ = data_quality_checkers.get_trigger_times(np.asarray(signals['camTrig']).reshape(-1), threshold=3.3)
            _, _, trig_num = data_quality_checkers.get_trigger_times(np.asarray(signals['camTrig'])[:(rising_edge_ids[0] + sample_num)].reshape(-1))
    
            if falling_edge_ids[0] < rising_edge_ids[0]:
                falling_edge_ids = falling_edge_ids[1:]
            # this condition implemented due to an unwanted spike (does not trigger frame acquisition) at the beginning of the trigger signal
            if (falling_edge_ids[0]-rising_edge_ids[0]) < 5:
                rising_edge_ids = rising_edge_ids[1:]
                falling_edge_ids = falling_edge_ids[1:]
                trig_num -= 1
            start_sample = rising_edge_ids[0]  # smr_start sample
        else:
            # consider only the first 250 frames (due to potentially delayed bonsai start times)
            trig_num = 500
            # this is ok because the exact timing of each frame is not important

            bodyX = dlc['body']['x'][:trig_num]
            bodyY = dlc['body']['y'][:trig_num]
            snoutX = dlc['snout']['x'][:trig_num]
            snoutY = dlc['snout']['y'][:trig_num]
            dlc_data[:, dlc_column] = utils_math.angle_with_x(snoutX, snoutY, bodyX, bodyY)
            dlc_tuples.append((mouse_id, level, trial, 'snoutBodyAngle'))

            weight, _ = get_weight_age(metadata_df, mouse_id, expDate=Config.expDates_config['head_height_exp'])
            if 'rl' in level:
                headHW_data[:, dlc_column] = ((-int(level[2:])+18.5)/Config.forceplate_config["mm_per_g"])/weight # was 17.5 for the fig 1 plots!!!
                # assume that a 22g mouse can just about comfortably reach the ground at rl-8 in this setting
                # and assume that the equivalent for rl-13 is a 26g mouse
                # x/22 = (x+5)/26, solve for x = 27.5, hence add 19.5 (-1*-8 + 19.5 = 27.5)
                headHW_tuples.append((mouse_id, level, trial, 'headHW'))

            dlc_column += 1

        # downsample if fps is higher than the lowest in the dataset (happens due to limited bandwidth with camTrig input channel)
        if fps > sample_num/Config.forceplate_setup["trial_duration"]:
            downsampling_ratio = fps/(sample_num/Config.forceplate_setup["trial_duration"])
            signals_downsampled = np.empty((int(signals.shape[0]/downsampling_ratio), len(Config.forceplate_config["corrected_labels"].values())))
            signals_downsampled = pd.DataFrame(signals_downsampled, columns=Config.forceplate_config["corrected_labels"].values())
            for limb in Config.forceplate_config["corrected_labels"].values():
                signals_downsampled[limb] = utils_processing.downsample_data(np.asarray(signals[limb]), int(signals.shape[0]/downsampling_ratio))
            start_sample = int(start_sample/downsampling_ratio)
        end_sample = start_sample + sample_num

        # add csv data to the dataframe
        for limb in Config.forceplate_config["corrected_labels"].values():
            csv_tuples.append((mouse_id, level, trial, limb))
            csv_data[:, csv_column] = signals[limb][start_sample: end_sample]
            csv_column += 1

    # if some signals are too short and are excluded, 'data' have empty columns at the end
    csv_data = csv_data[:, :len(csv_tuples)]
    dlc_data = dlc_data[:, :len(dlc_tuples)]
    if 'rl' in csv_tuples[0][1]:  # if head height experiment
        headHW_data = headHW_data[:, :len(headHW_tuples)]
        
    if yyyymmdd == None:
        from datetime import date
        yyyymmdd = str(date.today())  # '2021-10-26' #'2022-04-02'
    csv_index = pd.MultiIndex.from_tuples(csv_tuples, names=["mouse", "level", "trial", "limb"])
    csv_df = pd.DataFrame(csv_data, columns=csv_index)
    csv_df.to_csv(os.path.join(outputDir,  yyyymmdd + '_forceplateData.csv'))

    if type(dataDir) == str:
        dlc_index = pd.MultiIndex.from_tuples(dlc_tuples, names=["mouse", "level", "trial", "limb"])
        dlc_df = pd.DataFrame(dlc_data, columns=dlc_index)
        dlc_df.to_csv(os.path.join(outputDir, yyyymmdd + '_forceplateAngles.csv'))
        print(f"Failed DLC fraction: {failed_count/len(dlc_files)}")

        if 'rl' in csv_tuples[0][1]:  # if head height experiment
            hw_index = pd.MultiIndex.from_tuples(
                headHW_tuples, names=["mouse", "level", "trial", "limb"])
            headHW_df = pd.DataFrame(headHW_data, columns=hw_index)
            headHW_df.to_csv(os.path.join(outputDir, yyyymmdd + '_forceplateHeadHW.csv'))

def get_weight_age(metadata_df, mouseID, expDate=Config.forceplate_config["head_height_exp"]):
    """
    returns the weight and age of mouse when supplied metadataPyRAT_processed, mouseID, and expDate
    
    PARAMETERS:
        metadata_df (dataframe) : dataframe containing metadata from PyRAT
        mouseID (str) : mouse of interest
        expDate (int) : date of experiment int('yymmdd')
                                            
    RETURNS:
        weight (float) : animal weight on the supplied date (possibly interpolated)
        age (int) : animal age in days
    """
    metadata_sub = metadata_df.loc[metadata_df['mouseID']['mouseID'] == mouseID, :].iloc[:, 2:]
    metadata_sub = metadata_sub.loc[:, ~np.isnan(metadata_sub.iloc[0, :])]
    dates = np.unique(metadata_sub.columns.get_level_values(1)).astype(int)
    if int(expDate) in dates:
        weight = float(metadata_sub.loc[:, ('Weight', str(expDate))])
        age = int(metadata_sub.loc[:, ('Age', str(expDate))])
    else:
        from datetime import datetime
        expDate_rec = f'20{str(expDate)[:2]}-{str(expDate)[2:4]}-{str(expDate)[4:]}'
        expDate_dt = datetime.strptime(expDate_rec, "%Y-%m-%d")
        nextDate_id = np.where(dates > int(expDate))[0]
        if len(nextDate_id) > 0:
            nextDate = dates[nextDate_id[0]]
            previousDate = dates[nextDate_id[0]-1]

            nextDate_rec = f'20{str(nextDate)[:2]}-{str(nextDate)[2:4]}-{str(nextDate)[4:]}'
            previousDate_rec = f'20{str(previousDate)[:2]}-{str(previousDate)[2:4]}-{str(previousDate)[4:]}'

            nextDate_dt = datetime.strptime(nextDate_rec, "%Y-%m-%d")
            previousDate_dt = datetime.strptime(previousDate_rec, "%Y-%m-%d")

            day_frac_since_previous = (expDate_dt-previousDate_dt).days/(nextDate_dt-previousDate_dt).days

            nextWeight = float(metadata_sub.loc[:, ('Weight', str(nextDate))])
            previousWeight = float(metadata_sub.loc[:, ('Weight', str(previousDate))])
            weight = round(previousWeight + (day_frac_since_previous*(nextWeight-previousWeight)), 1)

        else:
            previousDate = dates[-1]
            previousDate_rec = f'20{str(previousDate)[:2]}-{str(previousDate)[2:4]}-{str(previousDate)[4:]}'
            previousDate_dt = datetime.strptime(previousDate_rec, "%Y-%m-%d")
            weight = float(metadata_sub.loc[:, ('Weight', str(dates[-1]))])

        previousAge = int(metadata_sub.loc[:, ('Age', str(previousDate))])
        age = previousAge + (expDate_dt-previousDate_dt).days

    return weight, age

def weight_calibration(yyyymmdd = '2022-05-25', outputDir=Config.paths["forceplate_output_folder"]):
    """
    loads load cell calibration files, fits a linear regression model to the data,
    saves its output parameters for use in data analysis
    
    PARAMETERS:
        yyyymmdd (str) : date of calibration
        outputDir (str) : folder for newly generated files
                                            
    WRITES:
        "{yyyymmdd}_load_cell_calibration_summary.csv" : parameters to reconstruct the fitted calibration equation
    
    """
    inv_label_dict = {v:k for k,v, in Config.forceplate_config['corrected_labels'].items()}
    calibration = {}
    
    if yyyymmdd == '2022-05-25':
        dataDir = r"Z:\murray\Zane\ForceSensors\calibration_new"
        weight_df = pd.read_csv(Path(outputDir) / f"{yyyymmdd}_load_cell_calibration.csv")        
        for f in weight_df['filename']:
            filepath = Path(dataDir) / (f + '.csv')
            if os.path.exists(filepath):
                signals = pd.read_csv(filepath)

            for limb in Config.forceplate_config['corrected_labels'].values():
                if limb not in calibration.keys():
                    calibration[limb] = {'weight': [], 'voltage': []}
                calibration[limb]['weight'].append(float(weight_df[weight_df['filename']==f][inv_label_dict[limb]]))
            
                calibration[limb]['voltage'].append(np.nanmean(signals[limb]))                            
                        
    elif yyyymmdd == '2020-06-22':
        weight_df = pd.read_csv(Path(outputDir) / f"{yyyymmdd}_load_cell_calibration.csv")
        calibration = {}
        for limb in Config.forceplate_config['corrected_labels'].values():
            weight_df_sub = weight_df[weight_df['Sensor'] == inv_label_dict[limb]]
            if limb not in calibration.keys():
                calibration[limb] = {'weight': [], 'voltage': []}
            calibration[limb]['weight'] =  np.asarray(weight_df_sub['weight'])
            calibration[limb]['voltage'] = np.asarray(weight_df_sub['voltage'])            
              
    from sklearn.linear_model import LinearRegression
    model_summary = pd.DataFrame(np.empty((2,4)), columns = Config.forceplate_config['corrected_labels'].values(), index = ['Intercept', 'Slope'])
    for limb in Config.forceplate_config['corrected_labels'].values():
        model = LinearRegression().fit(np.asarray(calibration[limb]['weight']).reshape(-1, 1), np.asarray(calibration[limb]['voltage']).reshape(-1, 1))
        model_summary.loc['Intercept', limb] = model.intercept_
        model_summary.loc['Slope', limb] = model.coef_[0]
           
    model_summary.to_csv(Path(outputDir)/f"{yyyymmdd}_load_cell_calibration_summary.csv")
    
def weight_calibrate_dataframe(df, weightCalib_df, metadata_df, limb_labels = Config.forceplate_config['corrected_labels']):
    """
    uses the calibration equation to convert mouse weights into sensor voltage
    PARAMETERS:
        df (multilevel dataframe) : e.g. dataToLoad = "forceplateData" with levels (mouse, level, trial, limb)
        weightCalib_df (2x4 dataframe) : df with intercepts and slopes (dataToLoad = 'weightCalibration')
        outputDir (dataframe) : dataToLoad = 'metadata_processed'
        limb_labels (list) : list correcting load cell-limb association
                                            
    RETURNS:
        df (multilevel dataframe) : the supplied df but modified to reflect the weights on each sensor, not voltage signals 
            for slope trials, this is the signal along the vertical axis (hypothenuse)
        headplate_df (multilevel dataframe) : weight fractions on the head fixation apparatus
        df_bodyweight_frac (multilevel dataframe): the supplied df but modified to reflect the weight distributions relative to total weight
    """
    for limb in list(limb_labels.values()):
        df.loc[:, (slice(None), slice(None), slice(None), limb)] = (df.loc[:, (slice(None), slice(None), slice(None), limb)]  - weightCalib_df.loc['Intercept', limb]) / weightCalib_df.loc['Slope', limb] 
        # this df now represents weight distribution detected by the load cells
    
    from copy import deepcopy
    df_bodyweight_frac = deepcopy(df)
    mice = np.unique(df.columns.get_level_values(0))
    headplate_df = df.loc[:, (slice(None), slice(None), slice(None), 'rF')]
    headplate_df.columns = pd.MultiIndex.from_tuples([(x,y,z,'headplate') for x,y,z,e in headplate_df.columns], names = headplate_df.columns.names) # change 'rF' to 'headplate'
    headplate_df[:] = np.nan
    for m in mice:
        weight, _ = get_weight_age(metadata_df, m, expDate=Config.forceplate_config['head_height_exp'])
        if 'deg' in headplate_df.columns.get_level_values(1)[0]: # slope trials
            degs = np.unique(headplate_df.columns.get_level_values(1))    
            for deg_str in degs:
                deg = deg_str[3:]
                df_bodyweight_frac.loc[:, (m, deg_str, slice(None), slice(None))] = df.loc[:, (m, deg_str, slice(None), slice(None))] / (weight / np.cos(abs(int(deg))*np.pi/180))
        else:    
            df_bodyweight_frac.loc[:, (m, slice(None), slice(None), slice(None))] = df.loc[:, (m, slice(None), slice(None), slice(None))] / weight 
    unique_trials = np.unique(np.asarray(df.columns.to_list())[:,:-1], axis = 0)
    for unt in unique_trials:
        detected_weight = np.sum(df_bodyweight_frac.loc[:, (unt[0],unt[1],unt[2], slice(None))], axis = 1)
        headplate_df.loc[:, (unt[0],unt[1],unt[2])] = 1-detected_weight
    # compute the sum of the four limbs for each trial (from the new df), then subtract that from the total weight
      
    return df, headplate_df, df_bodyweight_frac
    # return df_bodyweight_frac, headplate_df


        
        