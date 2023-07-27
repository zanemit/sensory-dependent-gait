"""
Code to perform basic quality checks on the raw data and organise them into an
easily readable format
"""
import numpy as np
import pandas as pd
from pathlib import Path
import os
import cv2
import pickle
import warnings
import sys
import scipy.stats

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")
from processing.data_config import Config
from processing.data_quality_checkers import *
from processing.data_loader import *
from processing.utils_processing import downsample_data, populate_nested_dict

def get_trials_and_optotriggers(trigger_array):
    """
    finds trial onsets and offsets
    
    ! a simplified version of the analogous corridor function because 
    these experiments had a maximum of one trigger !
    
    PARAMETERS:
        trigger_array (multi-d numpy array) : array with camera trigger trace in
                                            column 0 and optotrigger trace in column 1
                                            
    RETURNS:
        camera_triggers[trial_onsets] (1d array) : analog sample number corresponding to trial start (camera on)
        camera_triggers[trial_offsets] (1d array) : analog sample number corresponding to trial end (camera off)
        frame_num_per_trial (1d array) : number of camera frames per trial
    """
    camera_triggers, _, _ = get_trigger_times(trigger_array[:,0])
    interframe_int = scipy.stats.mode(np.diff(camera_triggers))[0][0] # assumes that most frames are acquired at a constant rate
    trial_onsets = np.concatenate(([0], np.where(np.diff(camera_triggers)>(interframe_int+10))[0]+1)) #+10 allows for a bit of jitter
    trial_offsets = np.concatenate((np.where(np.diff(camera_triggers)>(interframe_int+10))[0], [len(camera_triggers)-1])) #+10 allows for a bit of jitter
    frame_num_per_trial = trial_offsets-trial_onsets+1
    
    rising_opto_triggers, falling_opto_triggers, _ = get_trigger_times(trigger_array[:,1])
    if rising_opto_triggers.shape[0] == 0: #if no opto trigger
        stimType = 'noTrig'
        return camera_triggers[trial_onsets], camera_triggers[trial_offsets], frame_num_per_trial, False, False, stimType
    
    # determine stimulus type
    if len(rising_opto_triggers) == len(trial_onsets):
        stimType = 'tonic'        
    else: # what if the stimulus is a frequency pattern? must find the grand onset/offset first!
        stimType = str(int(scipy.stats.mode(rising_opto_triggers)[0][0])) + 'Hz'
        
    optoONframes_per_trial = np.zeros_like(trial_onsets)
    optoOFFframes_per_trial = np.zeros_like(trial_onsets)    
    # find video frames corresponding to stim on/off     
    for i in range(trial_onsets.shape[0]):
        mask_r = np.logical_and(rising_opto_triggers > camera_triggers[trial_onsets[i]], rising_opto_triggers <= camera_triggers[trial_offsets[i]]) # subsets rising_opto_ids for the particular trial!
        last_trig_before_optoON = camera_triggers[(camera_triggers >= camera_triggers[trial_onsets[i]]) & (camera_triggers < rising_opto_triggers[mask_r][0])][-1]
        optoONframes_per_trial[i] = np.where(camera_triggers == last_trig_before_optoON)[0][0] - np.where(camera_triggers == camera_triggers[trial_onsets[i]])[0][0]
        
        mask_f = np.logical_and(falling_opto_triggers > camera_triggers[trial_onsets[i]], falling_opto_triggers <= camera_triggers[trial_offsets[i]])
        if np.all(mask_f == False) and stimType == 'tonic': #most likely because the optotrig ends when frame acquisition is already done
            falling_opto_triggers[i] = camera_triggers[trial_offsets[i]-1]   
            mask_f = np.logical_and(falling_opto_triggers > camera_triggers[trial_onsets[i]], falling_opto_triggers <= camera_triggers[trial_offsets[i]])
        first_trig_after_optoOFF = camera_triggers[(camera_triggers > falling_opto_triggers[mask_f][-1]) & (camera_triggers <= camera_triggers[trial_offsets[i]])][0]
        optoOFFframes_per_trial[i] = np.where(camera_triggers == first_trig_after_optoOFF)[0][0] - np.where(camera_triggers == camera_triggers[trial_onsets[i]])[0][0]
    
    return camera_triggers[trial_onsets], camera_triggers[trial_offsets], frame_num_per_trial, optoONframes_per_trial, optoOFFframes_per_trial , stimType

def prescreen_file(hdfRPath, hdfLPath, csvPath, binPath):
    """
    checks that the number of camera triggers matches the number of frames
    finds the number of optogenetic stimuli

    PARAMETERS
    ----------
    hdfRPath (str) : full path to the right side .h5 data file
    hdfLPath (str) : full path to the left side .h5 data file
    csvPath (str) : full path to the .csv data file
    binPath (str) : full path to the .bin data file

    RETURNS
    -------
    valid (bool) : is the frame number correct?
    (optoON, optoOFF) (tuple) : optogenetic stimuli start/end frames
    numBP (int) : number of labelled bodyparts in DLC file

    """
    dlcDataR = pd.read_hdf(hdfRPath) # DataFrame (samples, 3*bodyparts)
    dlcDataL = pd.read_hdf(hdfLPath) # DataFrame (samples, 3*bodyparts)
    camData = pd.read_csv(csvPath)
    binData = np.fromfile(binPath, dtype = np.double).reshape(-1,7)
    
    # the following is needed because I sometimes had the patch cord attached and the trigger on but laser off
    # so the analog trigger signal alone cannot be used to determine stimType
    expDate = binPath.split('\\')[-1].split('_')[1] 
    if expDate in Config.analysis_config['stimulation_days']:
        optotrig = True
    else:
        optotrig = False
    
    ch1 = check_skipped_frames(camData.iloc[:,0], camera_path = csvPath)
    ch2 = check_skipped_triggers(camData.iloc[:,1], binData[:,0],bin_path = binPath, threshold = 3.3)
    ch3 = check_dlc_file(camData.iloc[:,0], dlcDataR, dlc_path = hdfRPath)
    ch4 = check_dlc_file(camData.iloc[:,0], dlcDataL, dlc_path = hdfLPath)
  
    if ch1 and ch2 and ch3 and ch4:
        trialON, trialOFF, frame_num_per_trial, optoONframes_per_trial, optoOFFframes_per_trial, stimType = get_trials_and_optotriggers(binData, optotrig)
        
        valid = []
        frac_bad = []
        fr_start = 0
        fr_end = 0
        for on, off, fr in zip(trialON, trialOFF, frame_num_per_trial):
            fr_end += fr
            treadmill_speed_trials = binData[on:off,2]
            treadmill_speed_trials_downsampled = downsample_data(treadmill_speed_trials, dlcDataR.loc[fr_start:fr_end].shape[0]) # frames (in a trial) at which treadmill moves
            dlcRbody_lkhd_arr = np.asarray(dlcDataR.loc[fr_start:fr_end, (slice(None), 'body', 'likelihood')][treadmill_speed_trials_downsampled > 0.1])
            dlcRsnout_lkhd_arr = np.asarray(dlcDataR.loc[fr_start:fr_end, (slice(None), 'snout', 'likelihood')][treadmill_speed_trials_downsampled > 0.1])
            dlcLlH1_lkhd_arr = np.asarray(dlcDataL.loc[fr_start:fr_end, (slice(None), 'lH1', 'likelihood')][treadmill_speed_trials_downsampled > 0.1])
            frac_badRbody = (dlcRbody_lkhd_arr < 0.95).sum() / dlcRbody_lkhd_arr.shape[0]
            frac_badRsnout = (dlcRsnout_lkhd_arr < 0.95).sum() / dlcRsnout_lkhd_arr.shape[0]
            frac_badLlH1 = (dlcLlH1_lkhd_arr < 0.95).sum() / dlcLlH1_lkhd_arr.shape[0]
            fr_start += fr
            if frac_badRbody < 0.1 and frac_badRsnout < 0.1 and frac_badLlH1 < 0.2:
                valid.append(1)
            else:
                valid.append(0)
            frac_bad.append((round(frac_badRbody,4), round(frac_badRsnout,4), round(frac_badLlH1,4)))
        
        valid = np.asarray(valid)
        # find sample IDs that match recorded frames
        trialIDs = np.concatenate([np.arange(tON,tOFF) for tON, tOFF in zip(trialON,trialOFF)])
        
        # downsample trial-relevant optotrig
        optotrig_trials = binData[trialIDs, 1]
        optotrig_trials_downsampled = downsample_data(optotrig_trials, dlcDataR.shape[0])
        
        # downsample trial-relevant trdm speed and identify trials with poor tracking of 'body'!
        treadmill_speed_trials = binData[trialIDs,2]
        treadmill_speed_trials_downsampled = downsample_data(treadmill_speed_trials, dlcDataR.shape[0])
       
        if not np.all(np.asarray(valid) == 1):
            print(f"Tracking not good enough in {1-(sum(valid)/len(valid))} trials!")
        return valid, frame_num_per_trial, frac_bad, treadmill_speed_trials_downsampled, optotrig_trials_downsampled, stimType, optoONframes_per_trial, optoOFFframes_per_trial
        
    else:
        print("At least one of the basic acquisition quality checks has failed!")
        valid = False
        return valid, False, False, False, False, False, False, False
    
def find_valid_data(dataDir,
                    outputDir = Config.paths["mtTreadmill_output_folder"],
                    adding = False,
                    yyyymmdd = None):
    """
    finds sets of h5, csv, avi x2, and bin files that pass the basic quality checks,
    such as camera frame number matching the number of triggers

    PARAMETERS
    ----------
    directory (str, optional) : path to data folder

    RETURNS
    -------
    dlcFilesR (list of strs) : names of "valid" DLC-R files
    dlcFilesL (list of strs) : names of "valid" DLC-L files
    optoBPdict (dict) : dictionary with DLC file names as keys and tuples of
                        optotrigger numbers and bodypart numbers as values
    optoTrigDict (dict) : nested dictionary with mouseID as key0, expDate as key1,
                        and tuples of (optoON, optoOFF, type) as values
    acqTimeDict (dict) : nested dictionary as above but with acquisition times in
                        computer clock seconds as values
    locomFrameDict (dict) : nested dictionary as above but with (locomON, locomOFF, belt_speed)
                        as values where belt_speed has been downsampled and sliced
                        to correspond to video frames
    invalid_dlcFs (dict) : dictionary with DLC file names as keys and fraction of 
                        bad (<0.95 likelihood) data as (DLC-R, DLC-L) tuples

    """
    dataDir = Path(dataDir).resolve()
    assert dataDir.exists(), "Directory does not exist!"
    allFiles = [f for f in os.listdir(dataDir)]
    dlcFilesR = [f for f in os.listdir(dataDir) if "videoR" in f and f.endswith('.h5')] # used because we only want to consider DLC-ed data
    dlcFilesR_firsts = [f for f in dlcFilesR if len(f.split('_')[4])==3] # subsets filenames that do not contain b,c,etc in TDx    
    dlcFilesR_lasts = np.setdiff1d(dlcFilesR, dlcFilesR_firsts) # subsets filenames that DO contain b,c,etc
    dlcFilesR = np.concatenate((dlcFilesR_firsts, dlcFilesR_lasts)) # recombines the two lists in a better order
    dlcFilesL = [f for f in os.listdir(dataDir) if "videoL" in f and f.endswith('.h5')]
    
    sr = Config.analysis_config["sample_rate"]
    fps = Config.analysis_config["fps"]
    limb_bodyparts = [bp for bp in Config.analysis_config["limb_bodyparts"] if '0' not in bp]
    other_bodyparts = Config.analysis_config["other_bodyparts"]
    max_trial_dur = Config.analysis_config["max_trial_duration"]
    volts_to_speed = Config.analysis_config["cm/s_per_V"]
    
    # prescreening files that have been processed by DLC
    if adding:
        optoTrigDict, _ = load_processed_data(outputDir, dataToLoad = 'optoTrigDict', userInput = userInput)
        
    else:
        optoTrigDict = {} # a dictionary of tuples (optoTrigONSETS, optoTrigOFFSETS, stimTYPE)
        
    invalid_dlcFs = {}
    
    # initialise array for downsampled treadmill speed data
    treadmillSpeeds = np.empty((fps*max_trial_dur, 0)) 
    treadmillSpeeds[:] = np.nan # filled with nans for easier removal of invalid columns later...
    
    # initialise array for downsampled optotrig data
    optoTrigs = np.empty((fps*max_trial_dur, 0)) 
    optoTrigs[:] = np.nan # filled with nans for easier removal of invalid columns later...
    
    # initialise DLC arrays
    dlcLimbs = np.empty((fps*max_trial_dur, 0 )) # each trial should have been 20s longbut there is some jitter in the closed loop bonsai/1401 system 
    dlcLimbs[:] = np.nan
    dlcOther = np.empty((fps*max_trial_dur, 0 )) # each trial should have been 20s longbut there is some jitter in the closed loop bonsai/1401 system 
    dlcOther[:] = np.nan
    
    tuples = []
    dlcLimb_tuples = []
    dlcOther_tuples = []
    for icol, dlcFR in enumerate(dlcFilesR):
        print(f"Preprocessing {dlcFR}...")
        filenameBase = dlcFR.split("videoR")[0]
        dlcFL = [f for f in dlcFilesL if filenameBase in f]
        if len(dlcFL)>0:
            dlcFL = dlcFL[0]
        else:
            print(f"ATTENTION! {filenameBase} does not appear to have been DLC'd for videoL!")
            continue
        paths = {'hdfR': os.path.join(dataDir, dlcFR)} # done this way because we want to link all data from the same trial
        paths['hdfL']  = os.path.join(dataDir, dlcFL)
        
        # parsing the file name
        mouseID, expDate, degDur = get_metadata_from_filename_mt(dlcFR)
        
        # define the exact paths to all associated files
        fileNotFoundCount = 0
        for key, appendix in zip(['csv', 'aviR', 'aviL', 'bin'], ['camera.csv', 'videoR.avi', 'videoL.avi', 'analog.bin']):
            fullName = filenameBase + appendix
            paths[key] = os.path.join(dataDir, fullName)
            if fullName not in allFiles:
                print(f'File {fullName} cannot be found!')
                fileNotFoundCount += 1
         
        if fileNotFoundCount == 0:
            valid, frame_num_per_trial, fracBadDLC, treadmill_speed_downsampled, optotrig_trials_downsampled, stimType, optoONframes_per_trial, optoOFFframes_per_trial = prescreen_file(paths['hdfR'], paths['hdfL'], paths['csv'], paths['bin'])
            
            if np.any(np.asarray(valid) == 1):
                # load DLC data
                print(f"Processing {dlcFR}")
                filenameBase = dlcFR.split("videoR")[0]
                dlcFL = [f for f in dlcFilesL if filenameBase in f][0]
                pathR = os.path.join(dataDir, dlcFR)
                pathL = os.path.join(dataDir, dlcFL)
                warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
                dlcTrackingsR, _ = preprocess_dlc_data(pathR) #dlcTrackings is a dictionary
                dlcTrackingsL, _ = preprocess_dlc_data(pathL)
                
                # initialise arrays (one per file, all trials)
                treadmillSpeed_array = np.empty((fps*max_trial_dur, valid.sum()))
                optoTrig_array = np.empty((fps*max_trial_dur, valid.sum()))
                dlcLimb_array = np.empty((fps*max_trial_dur, len(limb_bodyparts)*valid.sum()*2)) # each trial should have been 20s longbut there is some jitter in the closed loop bonsai/1401 system 
                dlcOther_array = np.empty((fps*max_trial_dur, len(other_bodyparts)*valid.sum()*2))
                
                # add nans
                treadmillSpeed_array[:] = np.nan
                optoTrig_array[:] = np.nan
                dlcLimb_array[:] = np.nan  
                dlcOther_array[:] = np.nan
                
                # initialise trial counter by checking if the particular mouse has data in another file on the same day
                if len([tp for tp in tuples if mouseID in tp and expDate in tp])>0: # find identical mouse+expData combos in the existing df
                # if fileNotFirst:
                    trialCount = np.max([tup[2] for tup in tuples if mouseID in tup and expDate in tup])+1 
                else:
                    trialCount = 0
                
                column = 0
                dlcLimb_column = 0
                dlcOther_column = 0
                frON = 0
                frOFF = 0
                
                for i in range(len(valid)):
                    frOFF += frame_num_per_trial[i]
                    if valid[i]:      
                        tuples.append((mouseID, expDate, trialCount, degDur, stimType))
                        frame_num = frame_num_per_trial[i]
                        treadmillSpeed_array[:frame_num, column] = treadmill_speed_downsampled[frON:frOFF] * volts_to_speed
                        optoTrig_array[:frame_num, column] = optotrig_trials_downsampled[frON:frOFF]
                        
                        for bp in limb_bodyparts:
                            for coord in ['x','y']:
                                dlcLimb_tuples.append((mouseID, expDate, trialCount, degDur, stimType, bp, coord))
                                if bp in ['lH1', 'lH2', 'lF1', 'lF2']:
                                    if coord == 'x': # invert and translate!
                                        dlcLimb_array[:frame_num, dlcLimb_column] = dlcTrackingsL[bp][coord][frON:frOFF]*-1+1200
                                    else: # coord == y, apppend dlcL as is
                                        dlcLimb_array[:frame_num, dlcLimb_column] = dlcTrackingsL[bp][coord][frON:frOFF]
                                else: # append dlcR
                                    dlcLimb_array[:frame_num, dlcLimb_column] = dlcTrackingsR[bp][coord][frON:frOFF]
                                dlcLimb_column += 1
                        for bp in other_bodyparts:
                            for coord in ['x','y']:
                                dlcOther_tuples.append((mouseID, expDate, trialCount, degDur, stimType, bp, coord))
                                dlcOther_array[:frame_num, dlcOther_column] = dlcTrackingsR[bp][coord][frON:frOFF]
                                dlcOther_column += 1
                        column += 1
                        trialCount += 1
                        
                    else: # the specific trial has poor tracking (valid[i] == 0)
                        print(f"Trial {i} deemed invalid in prescreening!")
                        invalid_dlcFs[dlcFR.split('video')[0] + "trial" + str(i)] = fracBadDLC[i]
                        
                    frON += frame_num_per_trial[i]
                    
                treadmillSpeeds = np.hstack((treadmillSpeeds, treadmillSpeed_array))
                optoTrigs = np.hstack((optoTrigs, optoTrig_array))
                dlcLimbs = np.hstack((dlcLimbs, dlcLimb_array))
                dlcOther = np.hstack((dlcOther, dlcOther_array))
                
                if type(optoONframes_per_trial) != bool:
                    optoTrigDict = populate_nested_dict(optoTrigDict, (optoONframes_per_trial[np.asarray(valid).astype(bool)], optoOFFframes_per_trial[np.asarray(valid).astype(bool)]), (mouseID, expDate, degDur))
                else:
                    optoTrigDict = populate_nested_dict(optoTrigDict, (optoONframes_per_trial, optoOFFframes_per_trial), (mouseID, expDate, degDur))
            
        else:
            print("fileNotFoundCount nonzero!")
            invalid_dlcFs[dlcFR.split('video')[0]] = fracBadDLC
    
    index = pd.MultiIndex.from_tuples(tuples, names = ["mouseID", "expDate", "trial", "degDur", "stimType"])
    dlcLimb_index = pd.MultiIndex.from_tuples(dlcLimb_tuples, names = ["mouseID", "expDate", "trial", "degDur", "stimType", "bodypart", "coord"])
    dlcOther_index = pd.MultiIndex.from_tuples(dlcOther_tuples, names = ["mouseID", "expDate", "trial", "degDur", "stimType", "bodypart", "coord"])
    treadmillSpeed_df = pd.DataFrame(treadmillSpeeds, columns = index)
    optoTrigs_df = pd.DataFrame(optoTrigs, columns = index)
    dlcLimbs_df = pd.DataFrame(dlcLimbs, columns = dlcLimb_index)
    dlcOther_df = pd.DataFrame(dlcOther, columns = dlcOther_index)
    
    return optoTrigDict, treadmillSpeed_df, optoTrigs_df, dlcLimbs_df, dlcOther_df, invalid_dlcFs

def summarise_data(dataDir, 
                   outputDir = Config.paths["mtTreadmill_output_folder"],
                   adding = False,
                   yyyymmdd = None
                   ): 
    """
    1. finds sets of h5, avi, csv, bin files that pass basic quality checks
    2. writes data to files with today's date

    PARAMETERS
    ----------
    dataDir (str, optional) : path to data folder
    outputDir (str, optional) : path to output folder
    adding (bool, optional) : add to an existing df if True, create a new one if False
    yyyymmdd (str, optional) : date of experiment

    RETURNS
    -------
    None, but writes files

    """
    if adding:
        if yyyymmdd is not None: 
            old_dlcLimbs_df, _ = load_processed_data(outputDir, dataToLoad = 'mtLimbData', userInput = yyyymmdd)
        else:
            old_dlcLimbs_df, yyyymmdd = load_processed_data(outputDir, dataToLoad = 'mtLimbData')
            
        optoTrigDict, new_treadmillSpeed_df, new_optoTrigs_df, new_dlcLimbs_df, new_dlcOther_df, new_badTrackingDict = find_valid_data(outputDir = outputDir, dataDir = dataDir, adding=True, yyyymmdd = yyyymmdd) 
        
        old_badTrackingDict, _ = load_processed_data(outputDir, dataToLoad = 'badTrackingDict', userInput = yyyymmdd)
        badTrackingDict = {**old_badTrackingDict, **new_badTrackingDict}
        
        old_treadmillSpeed_df, _ = load_processed_data(outputDir, dataToLoad = 'treadmillSpeed', userInput = yyyymmdd)
        treadmillSpeed_df = pd.concat([old_treadmillSpeed_df, new_treadmillSpeed_df], axis = 1)
        
        old_optoTrigs_df, _ = load_processed_data(outputDir, dataToLoad = 'optoTriggers', userInput = yyyymmdd)
        optoTrigs_df = pd.concat([old_optoTrigs_df, new_optoTrigs_df], axis = 1)
        
        old_dlcOther_df, _ = load_processed_data(outputDir, dataToLoad = 'mtOtherData', userInput = yyyymmdd)
        dlcOther_df = pd.concat([old_dlcOther_df, new_dlcOther_df], axis = 1)
        
        dlcLimbs_df = pd.concat([old_dlcLimbs_df, new_dlcLimbs_df], axis = 1)  
        
    else:
        optoTrigDict, treadmillSpeed_df, optoTrigs_df, dlcLimbs_df, dlcOther_df, badTrackingDict = find_valid_data(outputDir = outputDir, dataDir = dataDir, adding=False) 
        if yyyymmdd == None:
            from datetime import date
            yyyymmdd = str(date.today())
    
    dlcLimbs_df.to_csv(os.path.join(outputDir, yyyymmdd + '_motoTreadmillLimbData.csv'))
    dlcOther_df.to_csv(os.path.join(outputDir, yyyymmdd + '_motoTreadmillOtherData.csv'))
    treadmillSpeed_df.to_csv(os.path.join(outputDir, yyyymmdd + '_treadmillSpeed.csv'))
    optoTrigs_df.to_csv(os.path.join(outputDir, yyyymmdd + '_optoTriggers.csv')) 
    pickle.dump(badTrackingDict, open(os.path.join(outputDir, yyyymmdd+ '_badTrackingDict.pkl'), "wb" ))
    pickle.dump(optoTrigDict, open(os.path.join(outputDir, yyyymmdd+ '_optoTrigDict.pkl'), "wb"))
    
def get_metadata_from_filename_mt(filename):
    """
    returns experimental metadata from files named "ZM_date_mouse_freq_stimDur_headLVL_[..]"

    PARAMETERS
    ----------
    filename (str or Path variable) : name of the file

    RETURNS
    -------
    mouseID (str) : 'BAA1098955'
    expDate (str) : e.g. '210325'
    file_id (str) : '1' if there is 'b' in filename

    """
    fsplit = str(filename).split('_')
    mouseID = fsplit[2]
    expDate = fsplit[1]
    degDur = fsplit[3]

    return mouseID, expDate, degDur

def get_metadata_from_df(df, col):
    """
    returns experimental metadata from multilevel dataframes
    containing mouse id, experimental date, stim frequency, head level
    given the column number in the df
    
    PARAMETERS
    ----------
    df (multilevel dataframe)
    col (int) : dataframe column to fetch metadata for

    RETURNS
    -------
    mouse_id (str) : 'BAA1098955'
    expDate (str) : e.g. '210325'

    """
    if len(df.columns.names) == 1:
        print("Only one level detected in the dataframe! Function aborted!")
        return None
    mouseID = df.columns.get_level_values(0)[col]
    expDate  = df.columns.get_level_values(1)[col]
    trialNum  = df.columns.get_level_values(2)[col]
    trialType = df.columns.get_level_values(3)[col]
    stimType = df.columns.get_level_values(4)[col]
    
    return mouseID, expDate, trialNum, trialType, stimType