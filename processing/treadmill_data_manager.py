"""
Code to perform basic quality checks on the raw data and organise them into an
easily readable format
"""
import numpy as np
import pandas as pd
from pathlib import Path
import os
# import cv2
import pickle
import warnings
import sys

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")
from processing.data_config import Config
from processing.data_quality_checkers import *
from processing.data_loader import *
from processing.utils_processing import populate_nested_dict , preprocess_dlc_data
    
def organise_all_data(dataDir, yyyymmdd, trialType = None):
    """  
    PARAMETERS:
       dataDir (str) : path to a folder containing ONLY subfolders with raw data
                                            
    WRITES:
    -------
    '{yyyymmdd}_passiveOptoData{appdx}.csv' : a multi-level dataframe of DLC
                                            tracking data in the time window 
                                            [optostim_on-0.5sec, optostim_off+0.5sec]
    '{yyyymmdd}_beltSpeedData{appdx}.csv' : a multi-level dataframe of belt speed
                                            over the same time window as above
    '{yyyymmdd}_badTrackingDict{appdx}.pkl') : a list of files rejected due to
                                            poor tracking (mostly) or problems
                                            with data acquisition (rarely)
    """
    folders = [x for x in os.listdir(dataDir) if len(x) == 6]
    
    if yyyymmdd == None:
        from datetime import date
        yyyymmdd = str(date.today())
        
    for i, folder in folders:
        path = Path(dataDir)/folder
        if i == 0:
            summarise_data(path, adding = False, yyyymmdd = yyyymmdd, trialType = trialType)
        else:
            summarise_data(path, adding = True, yyyymmdd = yyyymmdd, trialType = trialType)

def get_opto_triggers(trigger_array): 
    """
    finds the rising/falling edges of the optogenetic trigger voltages
    and matches them to camera trigger frames & ultimately trigger numbers
    
    PARAMETERS:
        trigger_array (multi-d numpy array) : array with camera trigger trace in
                                            column 0 and optotrigger trace in column 1
                                            
    RETURNS:
        onset_frame (int) : camera frame during which opto stim comes on
        offset_frame (int) : camera frame during which opto stim turns off
        onset (int) : DAQ sample id of opto stim coming on
        offset (int) : DAQ sample id of opto stim turning off
        
        in case there is more than one trigger, only the first one is returned!
        (there should never be more than one trigger)
    """
    camera_triggers = get_trigger_times(trigger_array[:,0])[0]
    
    rising_opto_triggers = get_trigger_times(trigger_array[:,1])[0]
    if rising_opto_triggers.shape[0] == 0: #if no trigger
        return False, False, False, False
    onset = rising_opto_triggers[0] # optotrig onset in the analog file
    onset_frame = np.where(camera_triggers == camera_triggers[camera_triggers < onset][-1])[0][0]
    
    falling_opto_triggers = get_trigger_times(trigger_array[:,1])[1]
    offset = falling_opto_triggers[-1]
    offset_frame = np.where(camera_triggers == camera_triggers[camera_triggers< offset][-1])[0][0]
    
    return onset_frame, offset_frame, onset, offset

def get_locom_bouts(trigger_array): #trigger_array is already just the optoON window + 0.5s at each end
    """
    finds the onsets/offsets of locomotor bouts (>10 cm/s belt speed)
    and downsamples belt speed in the analysis window
    
    PARAMETERS:
        trigger_array (multi-d numpy array) : array with camera trigger trace in
                                            column 0, optotrigger trace in column 1,
                                            belt speed in column 2, rows correspond
                                            to the analysis window only!
                                            
    RETURNS:
        locom_ON (1d numpy array) : camera frame(s) during which locomotion starts FOR THE [optoON-0.5s, optoOFF+0.5s] WINDOW!!!
        locom_OFF (1d numpy array) : camera frame(s) during which locomotion ends FOR THE [optoON-0.5s, optoOFF+0.5s] WINDOW!!!
        belt_speed_resampled (1d numpy array) : belt speed in the analysis window
    """    
    fps = Config.passiveOpto_config['fps']    
    sr = Config.passiveOpto_config['sample_rate']
    
    belt_speed = trigger_array[:,2]
    belt_speed_filtered = butter_filter(belt_speed, filter_freq = 10, sampling_freq = sr) # oversmoothing for rough estimation
    # trig_rising_edge_id, _, trig_num = get_trigger_times(trigger_array[:,0])
    belt_speed_filtered_resampled = resample(belt_speed_filtered, int(fps*6))
    belt_speed_resampled = resample(belt_speed, int(fps*6))
    
    is_moving = (belt_speed_filtered_resampled > 0.1).astype(int)
    locomotion_diff = np.diff(is_moving)
    locom_ON = np.where(locomotion_diff == 1)[0]
    locom_OFF = np.where(locomotion_diff == -1)[0]
    
    # cleaning up locomotion detection  
    if is_moving.sum() == 0: # the mouse does not move at all
        return False, False, belt_speed_resampled
    if len(locom_ON) == 0: # mouse is moving when trial starts and does not restart
        locom_ON = np.asarray([0])
    if len(locom_OFF) > 0:
        if locom_OFF[0] == 0: # can happen
            locom_OFF = locom_OFF[1:]
    if len(locom_OFF) == 0: # mouse does not stop during the trial
        locom_OFF = np.asarray([int(6*fps)-1])
        
    if locom_OFF[0] < locom_ON[0]: # the mouse is moving when the video starts
        locom_ON = np.insert(locom_ON, 0, 0)
    if locom_OFF[-1] < locom_ON[-1]: # the mouse is moving when the video ends
        locom_OFF = np.append(locom_OFF, int(6*fps)-1)
    
    ons_to_be_deleted = []; offs_to_be_deleted = []
    if len(locom_ON) > 1:
        for i, ons in enumerate(locom_ON[:-1]):
            if locom_ON[i+1] - locom_OFF[i] < 100: # less than 1/4th of a second 'stopping':
                ons_to_be_deleted.append(i+1)
                offs_to_be_deleted.append(i)
    locom_ON = np.delete(locom_ON, ons_to_be_deleted)
    locom_OFF = np.delete(locom_OFF, offs_to_be_deleted)
    return locom_ON, locom_OFF, belt_speed_resampled

def summarise_data(dataDir, 
                   outputDir = Config.paths["passiveOpto_output_folder"], 
                   yyyymmdd = None, 
                   trialType = None,
                   adding = False): 
    """
    1. finds sets of h5, avi, csv, bin files that pass basic quality checks
    2. generates/loads 7-level tuples to uniquely identify xy DLC data
    3. produces a multilevel dataframe containing xy DLC data
    4. writes that dataframe to an .h5 file with today's date

    PARAMETERS
    ----------
    dataDir (str) : path to raw data folder
    outputDir (str, optional) : path to output folder
    yyyymmdd (str, optional) : date of experiment
    trialType (str or None) : 'incline' for incline trials, None for head height trials (default)
    adding (bool) : if True, adds data to existing datafiles

    WRITES (or updates - if adding == True)
    -------
    '{yyyymmdd}_passiveOptoData{appdx}.csv' : a multi-level dataframe of DLC
                                            tracking data in the time window 
                                            [optostim_on-0.5sec, optostim_off+0.5sec]
    '{yyyymmdd}_beltSpeedData{appdx}.csv' : a multi-level dataframe of belt speed
                                            over the same time window as above
    '{yyyymmdd}_badTrackingDict{appdx}.pkl' : a list of files rejected due to
                                            poor tracking (mostly) or problems
                                            with data acquisition (rarely)
    '{yyyymmdd}_locomFrameDict{appdx}.pkl' : a dict of locomotor bout onsets
                                            and offsets
    '{yyyymmdd}_optoTrigDict{appdx}.pkl' : a dict of optostim triggers                                     
    

    """
    if trialType != None: # incline trials
        appdx = f"_{trialType}"
    else:
        appdx = ""
    fps = Config.passiveOpto_config["fps"]
    
    if adding: # load existing data
        if yyyymmdd is not None:
            df, _ = load_processed_data(outputDir, dataToLoad = 'passiveOptoData', userInput = yyyymmdd, appdx = appdx)
        else:
            df, yyyymmdd = load_processed_data(outputDir, dataToLoad = 'passiveOptoData', appdx = appdx)
        
        dlcFilesR, dlcFilesL, optoBPdict, optoTrigDict, locomFrameDict, new_beltSpeed_df, new_badTrackingDict = find_valid_data(outputDir = outputDir, dataDir = dataDir, adding=True, yyyymmdd = yyyymmdd, appdx = appdx) 
    
        old_badTrackingDict, _ = load_processed_data(outputDir, dataToLoad = 'badTrackingDict', userInput = yyyymmdd, appdx = appdx)
        badTrackingDict = {**old_badTrackingDict, **new_badTrackingDict}
        
        old_beltSpeed_df, _ = load_processed_data(outputDir, dataToLoad = 'beltSpeedData', userInput = yyyymmdd, appdx = appdx)
        beltSpeed_df = pd.concat([old_beltSpeed_df, new_beltSpeed_df], axis = 1)
        
        oldTuples = [d[:4] for d in df.columns.to_list()]
        allTuples = get_tuple_backbones(dlcFilesR) # get 4-level tuples from all files
        
        # append data
        # compare the two tuples and add new data to the dataframe
        for (tup, dlcFR) in zip(allTuples, dlcFilesR):
            if tup not in oldTuples:
                print(f"Adding {dlcFR}")
                filenameBase = dlcFR.split("videoR")[0]
                dlcFL = [f for f in dlcFilesL if filenameBase in f]
                if len(dlcFL)>0:
                    dlcFL = dlcFL[0]
                else:
                    print(f"ATTENTION! {filenameBase} does not appear to have been DLC'd for videoL!")
                    continue
                pathR = os.path.join(dataDir, dlcFR)
                pathL = os.path.join(dataDir, dlcFL)
                warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
                dlcTrackingsR, bpsR = preprocess_dlc_data(pathR) #dlcTracking is a dictionary
                dlcTrackingsL, _ = preprocess_dlc_data(pathL)
                
                # add DLC data to the multilevel dataframe      
                trigFrame = optoTrigDict[tup[0]][tup[1]][tup[2]][tup[3]][0]
                for bp in bpsR:
                    for coord in ['x','y']:
                        fulltup = (tup[0], tup[1], tup[2], tup[3],  bp, coord)
                        if bp in ['lH1', 'lH2', 'lF1', 'lF2']:
                            if coord == 'x': # invert and translate!
                                df[fulltup]  = grab_data(dlcTrackingsL[bp][coord]*-1+1200, trigFrame-int(fps*0.5), trigFrame+int(fps*5.5))
                            else: # apppend dlcL as is
                                df[fulltup] = grab_data(dlcTrackingsL[bp][coord], trigFrame-int(fps*0.5), trigFrame+int(fps*5.5))
                        else:
                            df[fulltup] = grab_data(dlcTrackingsR[bp][coord], trigFrame-int(fps*0.5), trigFrame+int(fps*5.5))
        
    else:   
        dlcFilesR, dlcFilesL, optoBPdict, optoTrigDict,locomFrameDict, beltSpeed_df, badTrackingDict = find_valid_data(dataDir, appdx = appdx)
        tuples = []
        colNum = np.sum(np.product(np.array(list(optoBPdict.values())), axis = 1))
        data = np.empty((int(fps*6), int(colNum*2))) #0.5 s pre-stim, 5 s stim, 0.5 s post-stim = 6; *2 because of x,y
            
        #generate the massive multilevel df
        column = 0
        for dlcFR in dlcFilesR:
            print(f"Processing {dlcFR}")
            filenameBase = dlcFR.split("videoR")[0]
            dlcFL = [f for f in dlcFilesL if filenameBase in f][0]
            pathR = os.path.join(dataDir, dlcFR)
            pathL = os.path.join(dataDir, dlcFL)
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            dlcTrackingsR, bpsR = preprocess_dlc_data(pathR) #dlcTracking is a dictionary
            dlcTrackingsL, _ = preprocess_dlc_data(pathL)
            
            mouseID, expDate, stimFreq, headLVL = get_metadata_from_filename(dlcFR)
            
            trigFrame = optoTrigDict[mouseID][expDate][stimFreq][headLVL][0]
            for bp in bpsR:
                for coord in ['x','y']:
                    tuples.append((mouseID, expDate, stimFreq, headLVL, bp, coord))
                    if bp in ['lH1', 'lH2', 'lF1', 'lF2']:
                        if coord == 'x': # invert and translate!
                            data[:, column] = grab_data(dlcTrackingsL[bp][coord]*-1+1200, trigFrame-int(fps*0.5), trigFrame+int(fps*5.5))
                        else: # apppend dlcL as is
                            data[:, column] = grab_data(dlcTrackingsL[bp][coord], trigFrame-int(fps*0.5), trigFrame+int(fps*5.5))
                    else: # append dlcR
                        data[:, column] = grab_data(dlcTrackingsR[bp][coord], trigFrame-int(fps*0.5), trigFrame+int(fps*5.5))
                    column += 1
                  
        index = pd.MultiIndex.from_tuples(tuples, names = ["mouseID", "expDate", "stimFreq", "headLVL", "bodypart", "coord"])
        df = pd.DataFrame(data, columns=index)
    
        if yyyymmdd == None:
            from datetime import date
            yyyymmdd = str(date.today())
            
    df.to_csv(os.path.join(outputDir, yyyymmdd + f'_passiveOptoData{appdx}.csv'))
    beltSpeed_df.to_csv(os.path.join(outputDir, yyyymmdd + f'_beltSpeedData{appdx}.csv'))
    pickle.dump(badTrackingDict, open(os.path.join(outputDir, yyyymmdd+ f'_badTrackingDict{appdx}.pkl'), "wb" ))
    pickle.dump(locomFrameDict, open(os.path.join(outputDir, yyyymmdd+ f'_locomFrameDict{appdx}.pkl'), "wb"))
    pickle.dump(optoTrigDict, open(os.path.join(outputDir, yyyymmdd+ f'_optoTrigDict{appdx}.pkl'), "wb"))

def find_valid_data(dataDir,
                    outputDir = Config.paths["passiveOpto_output_folder"],
                    adding = False,
                    yyyymmdd = None,
                    appdx = None):
    """
    finds sets of h5, csv, avi (x2), and bin files that pass basic quality checks,
    such as camera frame number matching the number of triggers

    PARAMETERS
    ----------
    dataDir (str) : path to data folder
    outputDir (str, optional) : path to output folder
    adding (bool, optional) : add to an existing df if True, create a new one if False
    yyyymmdd (str, optional) : date of experiment
    appdx (str) : "" (empty) or "incline" for head height and incline trials respectively

    RETURNS
    -------
    dlcFilesR (list of strs) : names of "valid" DLC-R files
    dlcFilesL (list of strs) : names of "valid" DLC-L files
    optoBPdict (dict) : dictionary with DLC file names as keys and tuples of
                        optotrigger numbers and bodypart numbers as values
    optoTrigDict (dict) : nested dictionary with mouseID as key0, expDate as key1,
                        stimFreq as key3, and headLVL as key4, and tuples of 
                        (optoON, optoOFF) as values
    acqTimeDict (dict) : nested dictionary as above but with acquisition times in
                        computer clock seconds as values
    locomFrameDict (dict) : nested dictionary as above but with (locomON, locomOFF, belt_speed)
                        as values where belt_speed has been downsampled and sliced
                        to correspond to video frames
    invalid_dlcFs (dict) : dictionary with DLC file names as keys and fraction of 
                        bad (<0.95 likelihood) data as (frac_badRbody, frac_badRsnout, frac_badLlH1) tuples

    """
    dataDir = Path(dataDir).resolve()
    assert dataDir.exists(), "Directory does not exist!"
    allFiles = [f for f in os.listdir(dataDir)]
    dlcFilesR = [f for f in os.listdir(dataDir) if "videoR" in f and f.endswith('.h5')] # used because we only want to consider DLC-ed data  
    dlcFilesL = [f for f in os.listdir(dataDir) if "videoL" in f and f.endswith('.h5')]
    fps = Config.passiveOpto_config["fps"]
    sr = Config.passiveOpto_config['sample_rate']
    
    # prescreening files that have been processed by DLC
    if adding:
        locomFrameDict, _ = load_processed_data(outputDir, dataToLoad = 'locomFrameDict', yyyymmdd = yyyymmdd, appdx = appdx) 
        optoTrigDict, _ = load_processed_data(outputDir, dataToLoad = 'optoTrigDict', yyyymmdd = yyyymmdd, appdx = appdx)
        
    else:
        locomFrameDict = {} # a dictionary of frames where locomotion starts/ends in a given trial
        optoTrigDict = {} # a dictionary of tuples (optoTrigONSETS, optoTrigOFFSETS)
        
    invalid_dlcFs = {}
    optoBPdict = {} # a dictionary of tuples (optoTrigNum, bodypartNum)
    beltSpeeds = np.empty((int(fps*6), len(dlcFilesR)))
    beltSpeeds[:] = np.nan 

    tuples = []
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
        mouseID, expDate, stimFreq, headLVL = get_metadata_from_filename(dlcFR)
        
        # check that all files relevant to a particular trial are present
        fileNotFoundCount = 0
        for key, appendix in zip(['csv', 'bin'], ['camera.csv', 'analog.bin']):
            fullName = filenameBase + appendix
            paths[key] = os.path.join(dataDir, fullName)
            if fullName not in allFiles:
                print(f'File {fullName} cannot be found!')
                fileNotFoundCount += 1
                
        if fileNotFoundCount == 0:
            valid, optoTrigTimes, numBP, fracBadDLC, locomFrames, beltSpeed = prescreen_file(paths['hdfR'], paths['hdfL'], paths['csv'], paths['bin'])
            if valid:
                optoBPdict[dlcFR] = (1, numBP)
                
                # handle belt speed data, considering changes in Teensy voltage-speed relationship on 210604
                tuples.append((mouseID, expDate, stimFreq, headLVL))
                beltSpeeds[:,icol] = beltSpeed * 200 / 3.3
                
                optoTrigDict = populate_nested_dict(optoTrigDict, optoTrigTimes, (mouseID, expDate, stimFreq, headLVL))
                locomFrameDict = populate_nested_dict(locomFrameDict, locomFrames, (mouseID, expDate, stimFreq, headLVL))
            else:
                print("Set of files deemed invalid in prescreening!")
                invalid_dlcFs[dlcFR] = fracBadDLC
        else:
            print("fileNotFoundCount nonzero!")
            invalid_dlcFs[dlcFR] = fracBadDLC
        
    if len(invalid_dlcFs) > 0 :
        beltSpeeds = beltSpeeds[:, np.all(~np.isnan(beltSpeeds), axis = 0)]
        for inv in invalid_dlcFs.keys():
            dlcFilesR.remove(inv)
    
    index = pd.MultiIndex.from_tuples(tuples, names = ["mouseID", "expDate", "stimFreq", "headLVL"])
    beltSpeed_df = pd.DataFrame(beltSpeeds, columns = index)
    
    return dlcFilesR, dlcFilesL, optoBPdict, optoTrigDict, locomFrameDict, beltSpeed_df, invalid_dlcFs
        
def get_metadata_from_filename(filename):
    """
    returns experimental metadata from files named "ZM_date_mouse_freq_stimDur_headLVL_[..]"

    PARAMETERS
    ----------
    filename (str or Path variable) : name of the file

    RETURNS
    -------
    mouseID (str) : 'BAA1098955'
    expDate (str) : e.g. '210325'
    stimFreq (str) : e.g. '10mW'
    headLVL (str) : e.g. '50Hz'

    """
    fsplit = str(filename).split('_')
    mouseID = fsplit[2]
    expDate = fsplit[1]
    stimFreq = fsplit[3]
    headLVL = fsplit[5]
    return mouseID, expDate, stimFreq, headLVL

def prescreen_file(hdfRPath, hdfLPath, csvPath, binPath):
    """
    checks that the number of camera triggers matches the number of frames
    finds the number of optogenetic stimuli

    PARAMETERS
    ----------
    hdfPath (str) : full path to the .h5 data file
    csvPath (str) : full path to the .csv data file
    aviRPath (str) : full path to the videoR.avi data file
    aviLPath (str) : full path to the videoL.avi data file
    binPath (str) : full path to the .bin data file

    RETURNS
    -------
    valid (bool) : is the frame number correct?
    (optoON, optoOFF) (tuple) : optogenetic stimuli start/end frames
    numBP (int) : number of labelled bodyparts in DLC file
    (frac_badRbody, frac_badRsnout, frac_badLlH1) : fraction of data points that 
                                                    do not satisfy the 0.95 tracking 
                                                    likelihood threshold during
                                                    locomotion

    """
    dlcDataR = pd.read_hdf(hdfRPath) # DataFrame (samples, 3*bodyparts)
    dlcDataL = pd.read_hdf(hdfLPath) # DataFrame (samples, 3*bodyparts)
    camData = pd.read_csv(csvPath)
    binData = np.fromfile(binPath, dtype = np.double).reshape(-1,7)
    
    sr = Config.passiveOpto_config["sample_rate"]
    
    ch1 = check_skipped_frames(camData.iloc[:,0], camera_path = csvPath)
    ch2 = check_skipped_triggers(camData.iloc[:,1], binData[:,0], bin_path = binPath, threshold = 3.3)
    ch3 = check_dlc_file(camData.iloc[:,0], dlcDataR, dlc_path = hdfRPath)
    ch4 = check_dlc_file(camData.iloc[:,0], dlcDataL, dlc_path = hdfLPath)
    
    if ch1 and ch2 and ch3 and ch4:
        optoON, optoOFF, analogON, analogOFF = get_opto_triggers(binData)
        numBP = int(dlcDataR.shape[1]/3)
        locomON, locomOFF, belt_speed = get_locom_bouts(binData[analogON - int(sr*0.5) : analogON + int(sr*5.5), :])
        if type(locomON) == bool: # if the mouse does not move, accept the data based on tracking quality during opto stim
            is_moving = np.arange(optoON, optoOFF)
        else:
            is_moving = np.concatenate([np.arange(locomON[i],locomOFF[i]) for i in range(len(locomON))]) + optoON-(Config.passiveOpto_config['fps']/2)
            if (locomOFF[-1]+optoON-(Config.passiveOpto_config['fps']/2)) > dlcDataR.shape[0]:
                print("Acquisition stopped too soon relative to optoTRIG! Still accepting data...")
                is_moving = is_moving[is_moving < dlcDataR.shape[0]]
        
        if type(optoON) != bool:
            dlcRbody_lkhd_arr = np.asarray(dlcDataR.loc[is_moving, (slice(None), 'body', 'likelihood')])
            dlcRsnout_lkhd_arr = np.asarray(dlcDataR.loc[is_moving, (slice(None), 'snout', 'likelihood')])
            dlcLlH1_lkhd_arr = np.asarray(dlcDataL.loc[is_moving, (slice(None), 'lH1', 'likelihood')])
            frac_badRbody = (dlcRbody_lkhd_arr < 0.95).sum() / dlcRbody_lkhd_arr.shape[0]
            frac_badRsnout = (dlcRsnout_lkhd_arr < 0.95).sum() / dlcRsnout_lkhd_arr.shape[0]
            frac_badLlH1 = (dlcLlH1_lkhd_arr < 0.95).sum() / dlcLlH1_lkhd_arr.shape[0]
            if frac_badRbody < 0.1 and frac_badRsnout < 0.1 and frac_badLlH1 < 0.2:
                return True, (optoON, optoOFF), numBP, (frac_badRbody, frac_badRsnout, frac_badLlH1), (locomON, locomOFF), belt_speed
            else:
                print(f"Tracking not good enough! {(frac_badRbody*100):.0f}% DLC-R-body, {(frac_badRsnout*100):.0f}% DLC-R-snout, and {(frac_badLlH1*100):.0f}% DLC-L-lH1 are bad!")
                return False, False, False, (frac_badRbody, frac_badRsnout, frac_badLlH1), False, False
        else:
            print("Optotrigger not detected!")
            return False, False, False, False, False, False
    else:
        print("At least one of the basic acquisition quality checks has failed!")
        return False, False, False, False, False, False

def get_tuple_backbones(files):
    """
    produces tuples of mouse ID, date, stim_freq, and head level
    based on file names for all provided files

    PARAMETERS
    ----------
    files(list of strs) : list of DLC file names (that may or may not be
                            part of "valid" file sets, depending on whether
                            these files have been prescreened)

    RETURNS
    -------
    tuples (list of tuples) : contains the first four dataframe level identifiers

    """
    tuples = []
    for dlcF in files:        
        mouseID, expDate, stimFreq, headLVL = get_metadata_from_filename(dlcF)
             
        tuples.append((mouseID, expDate, stimFreq, headLVL))
    return tuples