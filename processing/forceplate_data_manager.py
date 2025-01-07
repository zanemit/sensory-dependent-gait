import os
import sys
import pandas as pd
import numpy as np
import neo
from pathlib import Path

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")
from processing.data_config import Config
from processing import data_loader, data_quality_checkers, process_metadata, utils_processing, utils_math

def preprocess_smr_file(filepath, 
                            label_dict, 
                            filter=False):
    """
    FOR DATA ACQUIRED BEFORE 2023
    converts spike2 output data into a dict = {ch1, ch2, ch3, ...}
    where each channel contains signal, timestamp, and sampling rate
    PARAMETERS:
        filepath (str) : full path to the Spike2 output file (.smr)
        label_dict (dict of str) : a dictionary that corrects channel labels 
                                    which were wrong because the mouse was 
                                    facing the other way
        filter (bool) : whether to apply a butterworth filter
    RETURNS:
        signals (dict) : dictionary containing signal, timestamp, and sampling rate organised by channel
    """
    yyyymmdd = str(filepath).split("_")[0][-6:]
    if int('20'+yyyymmdd[:2]) >= 2023:
        reader = neo.io.Spike2IO(filepath,  try_signal_grouping = False)  # read the block
    else:
        reader = neo.io.Spike2IO(filepath)  # read the block
    data = reader.read(lazy=False)[0] # used to have cascade=True

    signals = {}
    camTrig = {}
    for asig in data.segments[0].analogsignals:
        times = asig.times.rescale('s').magnitude  # extract sample times
        # determine channel name without the leading b'
        if int('20'+yyyymmdd[:2]) >= 2023:
            ch = str(asig.name).split(sep="'")[0]
        else:
            ch = str(asig.annotations['title']).split(sep="'")[1]
        fs = float(asig.sampling_rate)  # extract sampling frequency
        
        if int('20'+yyyymmdd[:2]) >= 2023:
            try:
                asig = asig.rescale('V').magnitude.reshape(1,-1)[0]
            except:
                continue
            
        if ch == 'camTrig':
            camTrig = {'signal': np.array(asig), 'time': times, 'fps': fs}
        if ch in label_dict.keys():  # only add the limb channels
            if filter:
                signal = utils_processing.butter_filter(
                    np.array(asig), filter_order=2, filter_freq=2, sampling_freq=fs)
            else:
                signal = np.array(asig)
            signals[label_dict[ch]] = {
                'signal': signal, 'time': times, 'fps': fs}
    return signals, camTrig

def get_data(dataDir = Config.paths["forceplate_data_folder"], 
             videoDir = Config.paths["forceplate_video_folder"],
             outputDir = Config.paths["forceplate_output_folder"], 
             yyyymmdd = None):
    """
    loads data from all .csv files in the given directory
    organises it in a 4-level dataframe (mouse-headlevel-trial-limb)
    
    THE OUTPUT FILES ARE RAW VOLTAGES, NON-CALIBRATED!

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
        dlc_files = [f for f in os.listdir(videoDir) if f.endswith('.h5')]
    else:
        raise ValueError("Invalid directory entered!")

    # mouse weights reorganised with the passiveOpto code and moved to the forceplate dir
    metadata_df = data_loader.load_processed_data(outputDir, dataToLoad='metadataProcessed')
    fps_db = pd.read_hdf(os.path.join(dataDir, 'fps_database.h5'))

    # check if all files have the same frame rate
    if not np.all(np.diff(fps_db["fps"].astype(float)) == 0):
        print(f'Spike2 sampling rate is variable! Found these values: {np.unique(fps_db["fps"].astype(float))} !')
    
    sample_num = int(float(fps_db["fps"].min())*Config.forceplate_config["trial_duration"]) # number of samples in 5 seconds
    print(f'Using a {sample_num/Config.forceplate_config["trial_duration"]} Hz frame rate')

    dlc_tuples = []
    dlc_data = np.empty((Config.forceplate_config['fps']*Config.forceplate_config["trial_duration"], len(dlc_files)))

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
        fps = fps_db[fps_db['filename'] == f[:-3] + 'smr']['fps'].iloc[0]
        if type(fps) == float:
            pass
        elif type(fps) != str:
            fps = float(fps.iloc[-1])
        else:
            fps = float(fps)

        # metadata
        fsplit = f.split('_')
        if int(yyyymmdd[:4])>= 2023:
            # the 2023/4 dataset includes experimental date too
            d_exp = fsplit[0]
            # d_exp = '20' + expDate[:2] + '-' + expDate[2:4] + '-' + expDate[4:]
            mouse_id = fsplit[1]
            level = fsplit[2]
            trial = fsplit[-1][0]
        else:
            if int(yyyymmdd[:4])== 2022:
                d_exp = Config.forceplate_config['incline_exp']
            elif int(yyyymmdd[:4])== 2021:
                d_exp = Config.forceplate_config['head_height_exp']
            mouse_id = fsplit[0]
            level = fsplit[1]
            trial = fsplit[-1][0]

        start_sample = 0

        # process videos
        f_dlc = [d for d in dlc_files if f[:-4] in d][0]
        path_dlc = os.path.join(videoDir, f_dlc)
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
        if int(yyyymmdd[:4])>=2023:
            dlc_tuples.append((d_exp, mouse_id, level, trial, 'snoutBodyAngle'))
        else:
            dlc_tuples.append((mouse_id, level, trial, 'snoutBodyAngle'))

        weight, _ = get_weight_age(metadata_df, mouse_id, expDate=d_exp)
        if 'rl' in level: # this was 18.5 (240610)
            headHW_data[:, dlc_column] = ((-int(level[2:])+24.5)/Config.forceplate_config["mm_per_g"])/weight # was 17.5 for the fig 1 plots!!!
            # assume that a 22g mouse can just about comfortably reach the ground at rl-8 in this setting
            # and assume that the equivalent for rl-13 is a 26g mouse
            # x/22 = (x+5)/26, solve for x = 27.5, hence add 19.5 (-1*-8 + 19.5 = 27.5)
            if '2021' in yyyymmdd:
                headHW_tuples.append((mouse_id, level, trial, 'headHW'))
            else:
                headHW_tuples.append((d_exp, mouse_id, level, trial, 'headHW'))

        dlc_column += 1

        # downsample if fps is higher than the lowest in the dataset (happens due to limited bandwidth with camTrig input channel)
        if fps > sample_num/Config.forceplate_config["trial_duration"]:
            downsampling_ratio = fps/(sample_num/Config.forceplate_config["trial_duration"])
            signals_downsampled = np.empty((int(signals.shape[0]/downsampling_ratio), len(Config.forceplate_config["corrected_labels"].values())))
            signals_downsampled = pd.DataFrame(signals_downsampled, columns=Config.forceplate_config["corrected_labels"].values())
            for limb in Config.forceplate_config["corrected_labels"].values():
                signals_downsampled[limb] = utils_processing.downsample_data(np.asarray(signals[limb]), int(signals.shape[0]/downsampling_ratio))
            start_sample = int(start_sample/downsampling_ratio)
        end_sample = start_sample + sample_num
        
        # add csv data to the dataframe
        for limb in Config.forceplate_config["corrected_labels"].values():
            if int(yyyymmdd[:4])>=2023:
                csv_tuples.append((d_exp, mouse_id, level, trial, limb))
            else:
                csv_tuples.append((mouse_id, level, trial, limb))
            csv_data[:, csv_column] = signals[limb][start_sample: end_sample]
            csv_column += 1

    # if some signals are too short and are excluded, 'data' have empty columns at the end
    csv_data = csv_data[:, :len(csv_tuples)]
    dlc_data = dlc_data[:, :len(dlc_tuples)]
    if 'rl' in csv_tuples[0][1] or 'rl' in csv_tuples[0][2]:  # if head height experiment
        headHW_data = headHW_data[:, :len(headHW_tuples)]
        
    if yyyymmdd == None:
        from datetime import date
        yyyymmdd = str(date.today())  # '2021-10-26' #'2022-04-02'
    if int(yyyymmdd[:4])>=2023:
        csv_index = pd.MultiIndex.from_tuples(csv_tuples, names=["expDate", "mouse", "level", "trial", "limb"])
    else:
        csv_index = pd.MultiIndex.from_tuples(csv_tuples, names=["mouse", "level", "trial", "limb"])
    csv_df = pd.DataFrame(csv_data, columns=csv_index)
    csv_df.to_csv(os.path.join(outputDir,  yyyymmdd + '_forceplateData.csv'))

    if type(dataDir) == str:
        if int(yyyymmdd[:4])>=2023:
            dlc_index = pd.MultiIndex.from_tuples(dlc_tuples, names=["expDate", "mouse", "level", "trial", "limb"])
        else:
            dlc_index = pd.MultiIndex.from_tuples(dlc_tuples, names=["mouse", "level", "trial", "limb"])
        dlc_df = pd.DataFrame(dlc_data, columns=dlc_index)
        dlc_df.to_csv(os.path.join(outputDir, yyyymmdd + '_forceplateAngles.csv'))
        print(f"Failed DLC fraction: {failed_count/len(dlc_files)}")

        if 'rl' in csv_tuples[0][1] or 'rl' in csv_tuples[0][2]:  # if head height experiment
            if int(yyyymmdd[:4])>=2023:
                hw_index = pd.MultiIndex.from_tuples(
                        headHW_tuples, names=["expDate", "mouse", "level", "trial", "limb"])
            else:
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
        weight = metadata_sub.loc[:, ('Weight', str(expDate))].iloc[0]
        age = metadata_sub.loc[:, ('Age', str(expDate))].iloc[0].astype(int)
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

            nextWeight = metadata_sub.loc[:, ('Weight', str(nextDate))].iloc[0]
            previousWeight = metadata_sub.loc[:, ('Weight', str(previousDate))].iloc[0]
            weight = round(previousWeight + (day_frac_since_previous*(nextWeight-previousWeight)), 1)

        else:
            previousDate = dates[-1]
            previousDate_rec = f'20{str(previousDate)[:2]}-{str(previousDate)[2:4]}-{str(previousDate)[4:]}'
            previousDate_dt = datetime.strptime(previousDate_rec, "%Y-%m-%d")
            weight = metadata_sub.loc[:, ('Weight', str(dates[-1]))].iloc[0]

        previousAge = metadata_sub.loc[:, ('Age', str(previousDate))].iloc[0].astype(int)
        age = previousAge + (expDate_dt-previousDate_dt).days

    return weight, age

def weight_calibration(yyyymmdd = '2022-05-25', 
                       inputType = 'csv', 
                       outputDir=Config.paths["forceplate_output_folder"]):
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
    
    if '2023-11' in yyyymmdd or '2023-12' in yyyymmdd or '2024-02' in yyyymmdd:
        yymmdd = "".join(yyyymmdd.split("-"))[2:]
        dataDir = Path(Config.paths['forceplate_data_folder'])/ f'calibration2023' / f'{yymmdd}_dcBNC_gain5000'
        calibfile = [f for f in os.listdir(outputDir) if 'load_cell_calibration.csv' in f and yyyymmdd[:4] in f]
        weight_df = pd.read_csv(Path(outputDir) / calibfile[0])        
        for f in weight_df['filename']:
            if inputType == 'smr':
                filepath = Path(dataDir) / (f + '.smr')
                if os.path.exists(filepath):
                    signals, camtrig = preprocess_smr_file(filepath, 
                                                           label_dict = Config.analysis_config[f'corrected_labels_new'], 
                                                           filter=False)
            elif inputType == 'csv':
                filepath = Path(dataDir) / (f + '.csv')
                if os.path.exists(filepath):
                    signals = pd.read_csv(filepath)
            weights = []
            voltages = []
            for limb in Config.analysis_config[f'corrected_labels_new'].values():
                if limb not in calibration.keys():
                    calibration[limb] = {'weight': [], 'voltage': []}
                calibration[limb]['weight'].append(float(weight_df[weight_df['filename']==f][inv_label_dict[limb]]))
            
                if inputType == 'smr':
                    calibration[limb]['voltage'].append(np.nanmean(signals[limb]['signal']))
                elif inputType == 'csv':
                    calibration[limb]['voltage'].append(np.nanmean(signals[limb]))  
                    
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
    
def weight_calibrate_dataframe(df, 
                               metadata_df,
                               yyyymmdd,
                               outputDir = Config.paths["forceplate_output_folder"],
                               limb_labels = Config.forceplate_config['corrected_labels']):
    """
    uses the calibration equation to convert mouse weights into sensor voltage
    PARAMETERS:
        df (multilevel dataframe) : e.g. dataToLoad = "forceplateData" with levels (mouse, level, trial, limb)
        weightCalib_df (2x4 dataframe) : df with intercepts and slopes (dataToLoad = 'weightCalibration')
        outputDir (dataframe) : dataToLoad = 'metadata_processed'
        exp_str = 'head_height_exp', 'egr3_exp', 'incline_exp'
        limb_labels (list) : list correcting load cell-limb association
                                            
    RETURNS:
        df (multilevel dataframe) : the supplied df but modified to reflect the weight distributions relative to total weight
        headplate_df (multilevel dataframe) : weight fractions on the head fixation apparatus
    """   
    if int(yyyymmdd[:4])==2022:
        d_calib = '2022-05-25'
        d_exp = Config.forceplate_config['incline_exp']
    elif int(yyyymmdd[:4])==2021:
        d_calib = '2020-06-22'
        d_exp = Config.forceplate_config['head_height_exp']
    
    if int(yyyymmdd[:4])>=2023:
        expDates = np.unique(df.columns.get_level_values(0))
        for d_exp in expDates:
            d_calib = '20' + d_exp[:2] + '-' + d_exp[2:4] + '-' + d_exp[4:]
            # import weight-voltage calibration files 
            weightCalib, _ = data_loader.load_processed_data(
                                outputDir = outputDir, 
                                dataToLoad = 'weightCalibration', 
                                yyyymmdd = d_calib)
            for limb in list(limb_labels.values()):
                df.loc[:, (d_exp, slice(None), slice(None), slice(None), limb)] = (df.loc[:, (d_exp, slice(None), slice(None), slice(None), limb)]  - weightCalib.loc['Intercept', limb]) / weightCalib.loc['Slope', limb] 
                # this df now represents weight distribution detected by the load cells
            
            df_d_exp_sub = df.loc[:, d_exp]
            mice = np.unique(df_d_exp_sub.columns.get_level_values(0))
            for m in mice:
                weight, _ = get_weight_age(metadata_df, m, expDate=d_exp)
                df.loc[:, (d_exp, m, slice(None), slice(None), slice(None))] = df.loc[:, (d_exp, m, slice(None), slice(None), slice(None))] / weight 
        
    else:
        weightCalib, _ = data_loader.load_processed_data(
                            outputDir = outputDir, 
                            dataToLoad = 'weightCalibration', 
                            yyyymmdd = d_calib)
        for limb in list(limb_labels.values()):
            df.loc[:, (slice(None), slice(None), slice(None), limb)] = (df.loc[:, (slice(None), slice(None), slice(None), limb)]  - weightCalib.loc['Intercept', limb]) / weightCalib.loc['Slope', limb] 
            # this df now represents weight distribution detected by the load cells
        mice = np.unique(df.columns.get_level_values(0))
        for m in mice:
            weight, _ = get_weight_age(metadata_df, m, expDate=d_exp) 
            if int(yyyymmdd[:4])==2022: # normalise by mouse weight x cosine of incline
                multipliers = np.array([np.cos(int(x[3:])*np.pi/180) for x in df.loc[:, (m, slice(None), slice(None), slice(None))].columns.get_level_values(1)])
                df.loc[:, (m, slice(None), slice(None), slice(None))] = df.loc[:, (m, slice(None), slice(None), slice(None))] / (weight *multipliers)
            
            else: # normalise by mouse weight
                df.loc[:, (m, slice(None), slice(None), slice(None))] = df.loc[:, (m, slice(None), slice(None), slice(None))] / weight 
        
    if df.columns.nlevels == 5:
         df.columns = df.columns.droplevel(0)
         
    headplate_df = df.loc[:, (slice(None), slice(None), slice(None), 'rF')]
    headplate_df.columns = pd.MultiIndex.from_tuples([(x,y,z,'headplate') for x,y,z,e in headplate_df.columns], names = headplate_df.columns.names) # change 'rF' to 'headplate'
    headplate_df.loc[:,:] = np.nan
    unique_trials = np.unique(np.asarray(df.columns.to_list())[:,:-1], axis = 0)
    for unt in unique_trials:
        detected_weight = np.sum(df.loc[:, (unt[0],unt[1],unt[2], slice(None))], axis = 1)
        headplate_df.loc[:, tuple(unt)] = 1-detected_weight
    # compute the sum of the four limbs for each trial (from the new df), then subtract that from the total weight
      
    return df, headplate_df
        
        