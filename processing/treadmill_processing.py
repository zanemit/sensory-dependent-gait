import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import copy
import itertools
import math
import scipy.signal

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_processing, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def get_body_angles(outputDir = Config.paths["passiveOpto_output_folder"],
                    treadmill_type = 'passive',
                    yyyymmdd = '2022-08-18',
                    appdx = None):
    """
    compute snout-body-xaxis and snout-body-tail angles for the whole interval 
    processed in passiveOptoData (0.5 s before and after optostim)
    
    PARAMETERS
    ----------
    outputDir (str, optional) : path to output folder
    treadmill_type (str) : permitted arguments are 'passive' and 'motorised'
    appdx (str) : "" (empty) or "incline" for head height and incline trials respectively
    
    WRITES (or updates - if adding == True)
    -------
    '{yyyymmdd}_bodyAngles{appdx}.csv' : snout-hump and snout-hump-tailbase angles in
                                        [optostim_on-0.5sec, optostim_off+0.5sec]
    """
    if appdx != None:
        appdx = f"_{appdx}"
    else:
        appdx = ""
    
    if treadmill_type == 'passive':
        data, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'passiveOptoData', appdx = appdx, yyyymmdd = yyyymmdd)
        data_relevant = data.loc[:, (slice(None), slice(None), slice(None), slice(None), ['snout', 'body', 'tailbase'])]
        tup_num = 4
        tuple_names = ["mouseID", "expDate", "stimFreq", "headLVL", "angle"]
    elif '2021' in yyyymmdd and treadmill_type == 'motorised':
        data_relevant, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'mtOtherData', yyyymmdd = yyyymmdd)
        tup_num = 4
        tuple_names = ["mouseID", "expDate", "trial", "stimType", "angle"]
    elif treadmill_type == 'motorised':
        data_relevant, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'mtOtherData', yyyymmdd = yyyymmdd)
        tup_num = 5
        tuple_names = ["mouseID", "expDate", "trial", "trialType", "stimType", "angle"]
        appdx = ""
    else:
        raise ValueError("Invalid treadmill type supplied! It can only be 'passive' or 'motorised'!")
        
    data_tuples = data_relevant.columns.values
    data_tuples = np.asarray([tup[:tup_num] for tup in data_tuples])
    data_tuples = np.unique(data_tuples, axis = 0)
    
    angles_array = np.empty((data_relevant.shape[0], data_relevant.shape[1]//2))*np.nan
    tuples = []
    for itup in range(data_tuples.shape[0]):
        if treadmill_type == 'passive':
            mouseID, expDate, stimFreq, headLVL = data_tuples[itup,:]
            data_relevant_sub = data_relevant[mouseID][expDate][stimFreq][headLVL]
        elif '2021' in yyyymmdd and treadmill_type == 'motorised':
            mouseID, expDate, trialNum, stimType = data_tuples[itup,:]
            data_relevant_sub = data_relevant[mouseID][expDate][trialNum][stimType]
        else: #motorised
            mouseID, expDate, trialNum, trialType, stimType = data_tuples[itup,:]
            data_relevant_sub = data_relevant[mouseID][expDate][trialNum][trialType][stimType]
        bodyX = data_relevant_sub['body']['x']
        bodyY = data_relevant_sub['body']['y']
        snoutX = data_relevant_sub['snout']['x']
        snoutY = data_relevant_sub['snout']['y']
        tailbaseX = data_relevant_sub['tailbase']['x']
        tailbaseY = data_relevant_sub['tailbase']['y']
           
        snoutBodyAngle = utils_math.angle_with_x(snoutX, snoutY, bodyX, bodyY)
        snoutBodyTailAngle = utils_math.angle_between_vectors_2d(bodyX, bodyY, snoutX, snoutY, bodyX, bodyY, tailbaseX, tailbaseY)
        tailBodyAngle = utils_math.angle_with_x(tailbaseX, tailbaseY,bodyX, bodyY)
        
        newcol = int(3*itup)
        for k, (arr, label) in enumerate(zip([snoutBodyAngle, snoutBodyTailAngle, tailBodyAngle], 
                                             ["snoutBody", "snoutBodyTail", "tailBodyAngle"])):
            if treadmill_type == 'passive':
                tuples.append((mouseID, expDate, stimFreq, headLVL, label))
            elif '2021' in yyyymmdd and treadmill_type == 'motorised':
                tuples.append((mouseID, expDate, trialNum, stimType, label))
            else:
                tuples.append((mouseID, expDate, trialNum, trialType, stimType, label))
            angles_array[:, newcol+k] = arr
     
    index = pd.MultiIndex.from_tuples(tuples, names = tuple_names)
    df = pd.DataFrame(angles_array, columns=index)
    
    df.to_csv(os.path.join(outputDir, yyyymmdd + f'_bodyAngles{appdx}.csv'))  
 
def get_relative_body_height(outputDir= Config.paths["passiveOpto_output_folder"], 
                             appdx = None, 
                             yyyymmdd = '2022-08-18'):
    import statistics
    if appdx != None:
        appdx = f"_{appdx}"
    else:
        appdx = ""
    if "Passive" in outputDir:
        data, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'passiveOptoData', appdx = appdx, yyyymmdd = yyyymmdd)
        mouse_speed,_ = data_loader.load_processed_data(outputDir, dataToLoad = 'beltSpeedData', appdx = appdx, yyyymmdd = yyyymmdd)
        
        d = 4
        tup_names = ['mouseID', 'expDate', 'stimFreq', 'headLVL']
        locom_threshold = 0.01 * 150/3.3
        
        data_body_y = data.loc[:, (slice(None), slice(None), slice(None), slice(None), 'body', 'y')]
        data_tailbase_y = data.loc[:, (slice(None), slice(None), slice(None), slice(None), 'tailbase', 'y')]
        data_rH1_y = data.loc[:, (slice(None), slice(None), slice(None), slice(None), 'rH1', 'y')]
        
        data_tuples = data_body_y.columns.values
        
    elif "Motorised" in outputDir:
        data_other, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'mtOtherData', appdx = appdx, yyyymmdd = yyyymmdd)
        data_limbs, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'mtLimbData', appdx = appdx, yyyymmdd = yyyymmdd)
        mouse_speed, _ = data_loader.load_processed_data(outputDir, appdx = appdx, dataToLoad = 'mouseSpeed', yyyymmdd = yyyymmdd)
        locom_threshold=5
        if '2021' in yyyymmdd:
            d = 4
            tup_names = ['mouseID', 'expDate', 'trial', 'stimType']
            
            data_body_y = data_other.loc[:, (slice(None), slice(None), slice(None), slice(None), 'body', 'y')]
            data_tailbase_y = data_other.loc[:, (slice(None), slice(None), slice(None), slice(None), 'tailbase', 'y')]
            data_rH1_y = data_limbs.loc[:, (slice(None), slice(None), slice(None), slice(None), 'rH1', 'y')]
        else:
            d = 5
            tup_names = ['mouseID', 'expDate', 'trial', 'trialType', 'stimType']
            
            data_body_y = data_other.loc[:, (slice(None), slice(None), slice(None), slice(None), slice(None), 'body', 'y')]
            data_tailbase_y = data_other.loc[:, (slice(None), slice(None), slice(None), slice(None), slice(None), 'tailbase', 'y')]
            data_rH1_y = data_limbs.loc[:, (slice(None), slice(None), slice(None), slice(None), slice(None), 'rH1', 'y')]
        data_tuples = mouse_speed.columns.values
    
    data_tuples = np.asarray([tup[:d] for tup in data_tuples]) # was :5 before incorporation of 'deg' level (5s for old data?)
    data_tuples = np.unique(data_tuples, axis = 0)
    
    body_relative = np.zeros_like(data_body_y) * np.nan
    tailbase_relative = np.zeros_like(data_tailbase_y) * np.nan
    new_tuples = []
    for c in range(mouse_speed.shape[1]):
        tup = mouse_speed.columns[c][:d]
        new_tuples.append(tup)
        body_y = np.asarray(data_body_y.loc[:,tup]).flatten()
        tailbase_y = np.asarray(data_tailbase_y.loc[:,tup]).flatten()
        rH1_y = np.asarray(data_rH1_y.loc[:,tup]).flatten()
        is_locomoting = np.asarray(mouse_speed.loc[:,tup]).flatten()>locom_threshold
        if "Motorised" in outputDir and '2021' not in yyyymmdd:
            is_locomoting = np.concatenate((is_locomoting, [False])) # for 2022-05-06/2023-09-25 mtTreadmill
        rH1_y_locom = utils_processing.remove_outliers(rH1_y[is_locomoting])
        if rH1_y_locom.shape[0] == 0:
            body_relative[:,c] = [np.nan] * body_relative.shape[0]
            tailbase_relative[:,c] = [np.nan] * tailbase_relative.shape[0]
        else:
            rH1_y_locom_mode = statistics.mode(rH1_y_locom)
            body_relative[:,c] = body_y-rH1_y_locom_mode  
            tailbase_relative[:,c] = tailbase_y-rH1_y_locom_mode 
    
    index = pd.MultiIndex.from_tuples(new_tuples, names = tup_names)
    body_relative_DF = pd.DataFrame(body_relative*-1, columns = index)
    body_relative_DF.to_csv(os.path.join(outputDir, yyyymmdd + f'_bodyHeight{appdx}.csv'))  
    
    tailbase_relative_DF = pd.DataFrame(tailbase_relative*-1, columns = index)
    tailbase_relative_DF.to_csv(os.path.join(outputDir, yyyymmdd + f'_tailbaseHeight{appdx}.csv'))  
   
def reformat_data(outputDir = Config.paths["passiveOpto_output_folder"], appdx = None):
    """
    uses the opto-ON DLC .csv file to generate a Tx8 array where T = the number of
    locomotor timepoints across mice (with nans removed) and the number of columns 
    refers to foot labels
    the array contains filtered angular velocities of those body labels
    
    PARAMETERS
    ----------
    outputDir (str, optional) : path to output folder
    appdx (str) : "" (empty) or "incline" for head height and incline trials respectively
    
    WRITES
    ----------
    '{yyyymmdd}_arrayIdentifier{appdx}.csv' : T x 4 array showing the file identity
                                            (mouse, date, stim_freq, head_lvl) of 
                                            each data point
    '{yyyymmdd}_limbX{appdx}.csv' : T x 8 array with raw x coordinates of limb body labels
    '{yyyymmdd}_limbY{appdx}.csv' : T x 8 array with raw y coordinates of limb body labels
    
    """
    if appdx != None:
        appdx = f"_{appdx}"
    else:
        appdx = ""
        
    data, yyyymmdd = data_loader.load_processed_data(outputDir, dataToLoad = 'passiveOptoData', appdx = appdx)
    locomFrameDict, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'locomFrameDict', yyyymmdd = yyyymmdd, appdx = appdx)
    speed, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'beltSpeedData', yyyymmdd = yyyymmdd, appdx = appdx)
    bodyAngles, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'bodyAngles', yyyymmdd = yyyymmdd, appdx = appdx)
    body_heights, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'bodyHeight', yyyymmdd = yyyymmdd, appdx = appdx)
    tailbase_heights, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'tailbaseHeight', yyyymmdd = yyyymmdd, appdx = appdx)
    
    
    labels = ['lH1', 'lH2', 'rH1', 'rH2', 'lF1', 'lF2', 'rF1', 'rF2']
    
    # initialise storage arrays
    data_relevant = data.loc[:, (slice(None), slice(None), slice(None), slice(None), labels)]
    array_ang = np.empty((0, len(labels)))
    array_x = np.empty((0, len(labels)))
    array_y = np.empty((0, len(labels)))
    array_speed = np.empty(0) # column = speed
    array_bodyAngles = np.empty((0, 3)) # columns = bodySnout, snoutBodyTail, tailBody
    array_bodyHeights_rel = np.empty((0))
    array_tailbaseHeights_rel = np.empty((0))
    mouse = []; date = []; freq = []; level = []
    
    # loop over mouseID-expDate-stimFreq-headLVL combinations to find the corresponding locomON/OFF and subset the data accordingly
    for mouseID in locomFrameDict.keys():
        for expDate in locomFrameDict[mouseID].keys():
            for stimFreq in locomFrameDict[mouseID][expDate].keys():
                for headLVL in locomFrameDict[mouseID][expDate][stimFreq].keys():
                    print(f'Processing {mouseID} {expDate} {stimFreq} {headLVL}...')
                    locomON = locomFrameDict[mouseID][expDate][stimFreq][headLVL][0]
                    locomOFF = locomFrameDict[mouseID][expDate][stimFreq][headLVL][1]
                    if type(locomON) == bool or type(locomOFF)==bool:
                        continue
                    locomIDs = [np.arange(on,off) for on, off in zip(locomON,locomOFF)]
                    locomIDs = utils_processing.flatten_list(locomIDs)
                    data_relevant_sub = data_relevant.loc[:, (mouseID, expDate, stimFreq, headLVL)]
                    data_relevant_sub = np.asarray(data_relevant_sub.iloc[locomIDs, :])
                    # data_relevant_sub = data_relevant_sub[~np.isnan(data_relevant_sub).any(axis=1), :] #since preprocess_dlc interpolates nans, nans can only be present at the start or end
                    data_relevant_sub = np.asarray(data_relevant_sub).reshape(-1, data_relevant_sub.shape[1]//2, 2) # separates x and y in separate dimensions
                    # data_relevant_sub = data_relevant_sub.reshape(data_relevant_sub.shape[0]*data_relevant_sub.shape[1], 2, order = 'F') # stacks body parts on top of each other
                    
                    angles_arr = np.zeros((data_relevant_sub.shape[0], data_relevant_sub.shape[1]))
                    xcoords_arr = np.zeros((data_relevant_sub.shape[0], data_relevant_sub.shape[1]))
                    ycoords_arr = np.zeros((data_relevant_sub.shape[0], data_relevant_sub.shape[1]))
                    for bp in range(data_relevant_sub.shape[1]):
                        # try:
                        angles = utils_math.temporal_angle(data_relevant_sub[:, bp, 0], data_relevant_sub[:, bp, 1])
                        xcoord = data_relevant_sub[:, bp, 0]
                        ycoord = data_relevant_sub[:, bp, 1]
                        if  angles[np.isnan(angles)].shape[0] > 0:
                            angle_nans = np.where(np.isnan(angles))[0]
                            if  angle_nans[0] == 1 and  angle_nans[-1] == (len(angles)-1):
                                print("Both leading and trailing nans! Both will NOT be corrected! Discarding {mouseID} {expDate} {stimFreq} {headLVL}...")
                                continue
                            elif  angle_nans[0] == 1:
                                print("Leading nans!")
                                first_nonnan_id = np.where(~np.isnan(angles))[0][1] # take id=1 becausea leading zero added to angles
                                angles[np.isnan(angles)] = np.repeat(angles[first_nonnan_id],  angle_nans.shape[0])
                                xcoord[np.isnan(xcoord)] = np.repeat(xcoord[first_nonnan_id-1],  angle_nans.shape[0])
                                ycoord[np.isnan(ycoord)] = np.repeat(ycoord[first_nonnan_id-1],  angle_nans.shape[0])
                            elif  angle_nans[-1] == (len(angles)-1):
                                print("Trailing nans!")
                                last_nonnan_id = np.where(~np.isnan(angles))[0][-1] 
                                angles[np.isnan(angles)] = np.repeat(angles[last_nonnan_id],  angle_nans.shape[0])
                                xcoord[np.isnan(xcoord)] = np.repeat(xcoord[last_nonnan_id-1],  angle_nans.shape[0])
                                ycoord[np.isnan(ycoord)] = np.repeat(ycoord[last_nonnan_id-1],  angle_nans.shape[0])
                            else:
                                print("Some nans in the middle! Discarding {mouseID} {expDate} {stimFreq} {headLVL}...")
                                continue
                        try:
                            angles_arr[:,bp] = utils_processing.butter_filter(angles, filter_freq = 5)
                            xcoords_arr[:,bp] = xcoord #butter_filter(xcord, filter_freq = 5)
                            ycoords_arr[:,bp] = ycoord
                        except:
                            print(f"{mouseID} {expDate} {stimFreq} {headLVL} failed to get angles; locomotion occurs over {data_relevant_sub.shape[0]} frames!")
                            continue
                        
                    speed_sub = np.asarray(speed.loc[locomIDs, (mouseID, expDate, stimFreq, headLVL)])
                    bodyAngles_sub = np.asarray(bodyAngles.loc[locomIDs, (mouseID, expDate, stimFreq, headLVL)])
                    body_heights_sub = np.asarray(body_heights.loc[locomIDs, (mouseID, expDate, stimFreq, headLVL)]).flatten()
                    tailbase_heights_sub = np.asarray(tailbase_heights.loc[locomIDs, (mouseID, expDate, stimFreq, headLVL)]).flatten()               
                    
                    # APPENDING
                    # array_ang = np.append(array_ang, angles_arr, axis = 0)
                    array_x = np.append(array_x, xcoords_arr, axis = 0)
                    array_y = np.append(array_y, ycoords_arr, axis = 0)
                    array_speed = np.append(array_speed, speed_sub)
                    array_bodyAngles = np.append(array_bodyAngles, bodyAngles_sub, axis = 0)
                    array_bodyHeights_rel = np.append(array_bodyHeights_rel, body_heights_sub, axis = 0)
                    array_tailbaseHeights_rel = np.append(array_tailbaseHeights_rel, tailbase_heights_sub, axis = 0)
                    mouse.append(np.repeat(mouseID, angles_arr.shape[0]))
                    date.append(np.repeat(expDate, angles_arr.shape[0]))
                    freq.append(np.repeat(stimFreq, angles_arr.shape[0]))
                    level.append(np.repeat(headLVL, angles_arr.shape[0]))
    
    mouse = utils_processing.flatten_list(mouse)
    date = utils_processing.flatten_list(date)
    freq = utils_processing.flatten_list(freq)
    level = utils_processing.flatten_list(level)
        
    array_identifier = np.vstack((mouse, date, freq, level)).T      
    np.save(Path(outputDir)/ (yyyymmdd + f"_arrayIdentifier{appdx}.npy"), array_identifier)   
    # np.save(Path(outputDir)/ (yyyymmdd + f"_limbAngVel{appdx}.npy"), array_ang)  
    np.save(Path(outputDir)/ (yyyymmdd + f"_limbX{appdx}.npy"), array_x)  #   'lH1', 'lH2', 'rH1', 'rH2', 'lF1', 'lF2', 'rF1', 'rF2'
    np.save(Path(outputDir)/ (yyyymmdd + f"_limbY{appdx}.npy"), array_y)  #   'lH1', 'lH2', 'rH1', 'rH2', 'lF1', 'lF2', 'rF1', 'rF2'
    np.save(Path(outputDir) / (yyyymmdd + f"_limbX_speed{appdx}.npy"), array_speed)
    np.save(Path(outputDir) / (yyyymmdd + f"_limbX_bodyAngles{appdx}.npy"), array_bodyAngles)
    np.save(Path(outputDir) / (yyyymmdd + f"_limbX_bodyHeights_rel{appdx}.npy"), array_bodyHeights_rel)
    np.save(Path(outputDir) / (yyyymmdd + f"_limbX_tailbaseHeights_rel{appdx}.npy"), array_tailbaseHeights_rel)

def compute_gait_params_from_xcoords(limb = 'lH1', 
                                     appdx = "", 
                                     outputDir = Config.paths["passiveOpto_output_folder"]):
    from scipy.stats import pearsonr
    if appdx != "":
        appdx = f"_{appdx}"
        
    arrayX, yyyymmdd = data_loader.load_processed_data(outputDir, 
                                                       dataToLoad = 'limbX', 
                                                       appdx = appdx)
    arrayY, _ = data_loader.load_processed_data(outputDir, 
                                                yyyymmdd = yyyymmdd,
                                                dataToLoad = 'limbY', 
                                                appdx = appdx)
    arrayIDer, _ = data_loader.load_processed_data(outputDir, 
                                                   dataToLoad = 'arrayIdentifier', 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = appdx)
    speed, _ = data_loader.load_processed_data(outputDir, 
                                               dataToLoad = 'limbX_speed', 
                                               yyyymmdd= yyyymmdd, 
                                               appdx = appdx)
    bodyAngles, _ = data_loader.load_processed_data(outputDir, 
                                                    dataToLoad = 'limbX_bodyAngles', 
                                                    yyyymmdd = yyyymmdd, 
                                                    appdx = appdx) #'snoutBody' 'snoutBodyTail', 'tailBody
    bodyHeights_rel, _ = data_loader.load_processed_data(outputDir, 
                                                         dataToLoad = 'limbX_bodyHeights_rel', 
                                                         yyyymmdd = yyyymmdd, 
                                                         appdx = appdx)
    tailbaseHeights_rel, _ = data_loader.load_processed_data(outputDir, 
                                                         dataToLoad = 'limbX_tailbaseHeights_rel', 
                                                         yyyymmdd = yyyymmdd, 
                                                         appdx = appdx)
    if os.path.exists(Path(outputDir)/"metadataPyRAT_processed.csv"):
        metadata_df = data_loader.load_processed_data(outputDir, 
                                                      dataToLoad = 'metadataProcessed', 
                                                      yyyymmdd = yyyymmdd, 
                                                      appdx = appdx)
        if 'rl' in arrayIDer[0,3]:
            headHeight_scaled = [(-int(x[2:])+24.5)/Config.passiveOpto_config["mm_per_g"] for x in arrayIDer[:,3]]
            # rl-8 is the max comfortable height for a 26g mouse
            # rl-3 is the max comfortable height for a 22g mouse
            # i.e. plugging in -3 or -8 for x will result in 22 and 26 respectively
            # when divided by mouse weight later, this will yield headHW = 1
    
    # filter the xcoord array to smoothen out tracking imprecisions
    print("Filtering the coord data...")
    arrayX_filtered = np.zeros_like(arrayX)
    arrayX_filtered2 = np.zeros_like(arrayX)
    arrayY_filtered = np.zeros_like(arrayY)
    for col in range(arrayX.shape[1]):
        arrayX_filtered[:,col] = utils_processing.butter_filter(arrayX[:,col], filter_freq = 2)
        arrayX_filtered2[:,col] = utils_processing.butter_filter(arrayX[:,col], filter_freq = 10)
        arrayY_filtered[:,col] = utils_processing.butter_filter(arrayY[:,col], filter_freq = 10)
    
    # find borders between files by comparing consecutive rows
    # each row of arrayIDer contains mouse-date-freq-level of a data point in array
    file_borders = [0]
    for i in range(arrayIDer.shape[0]-1):
        are_identical = (arrayIDer[i,:] == arrayIDer[i+1,:]).sum()
        if are_identical < 4:
            file_borders.append(i+1)
    file_borders.append(arrayIDer.shape[0]-1)   
    
    limb_id_dict = {'lH1': 0, 'lH2': 1, 'rH1': 2, 'rH2': 3, 'lF1': 4, 'lF2': 5, 'rF1': 6, 'rF2': 7} #cannot be changed because this is the order in the limbX file
    
    strideLength = np.empty((0))
    strideFreq = np.empty((0))
    mouseID = np.empty((0))
    expDate = np.empty((0))
    stimFreq = np.empty((0))
    headLVL = np.empty((0))
    strideNum = np.empty((0))
    sexes = np.empty((0))
    weights = np.empty((0))
    ages = np.empty((0))
    headHW = np.empty((0)) # head heights scaled by the average size of female/male mice
    lH1 = []; lH2 = []; lH0 = []
    rH1 = []; rH2 = []; rH0 = []
    lF1 = []; lF2 = []; lF0 = []
    rF1 = []; rF2 = []; rF0 = []
    speed_arr = []; snoutBody_arr = []; bodyAngleRange_arr = []
    bodyHeight_arr = []; tailbaseHeight_arr = []; tailBody_arr = []
    
    if limb == 'lH1':
        limb_list = [lH2, rH1, rH2, lF1, lF2, rF1, rF2]
        limb_list_str = ['lH2', 'rH1', 'rH2', 'lF1', 'lF2', 'rF1', 'rF2']
        mean_limb_list = [rH0, lF0, rF0] # for storage of mean of (rH1,rH2) etc
        mean_limb_str = ['rH0', 'lF0', 'rF0']
    elif limb == 'rH1':
        limb_list = [rH2, lH1, lH2, lF1, lF2, rF1, rF2]
        limb_list_str = ['rH2', 'lH1', 'lH2', 'lF1', 'lF2', 'rF1', 'rF2']
        mean_limb_list = [lH0, lF0, rF0] # for storage of mean of (lH1,lH2) etc
        mean_limb_str = ['lH0', 'lF0', 'rF0']
    elif limb == 'lF1':
        limb_list = [lF2, rH1, rH2, lH1, lH2, rF1, rF2]
        limb_list_str = ['lF2','rH1', 'rH2', 'lH1', 'lH2', 'rF1', 'rF2']
        mean_limb_list = [rH0, lH0, rF0] # for storage of mean of (lH1,lH2) etc
        mean_limb_str = ['rH0', 'lH0', 'rF0']
    
    stride_pt_max = Config.passiveOpto_config['fps']
    arrayX_strides = np.empty((stride_pt_max,0)) # for coordinate data per stride
    arrayX_strides2 = np.empty((stride_pt_max,0))
    arrayY_strides = np.empty((stride_pt_max,0))
    
    # consider each constituent file separately - ONLY lH1 or rH1 for now
    # use the filtered data to identify the number of peaks needed and find raw peaks in vicinity
    for fon, foff in zip(file_borders[:-1], file_borders[1:]):
        print(f"Working on {'_'.join(arrayIDer[fon,:])} between {fon} and {foff}...")
        arrX_filt_seg = arrayX_filtered[fon:foff, limb_id_dict[limb]]
        arrX_filt_seg2 = arrayX_filtered2[fon:foff, limb_id_dict[limb]]
        arrY_filt_seg = arrayY_filtered[fon:foff, limb_id_dict[limb]]
        arr_seg = arrayX[fon:foff, limb_id_dict[limb]]
        peaks_filt = scipy.signal.find_peaks(arrX_filt_seg, prominence = 50)[0]
        troughs_filt = scipy.signal.find_peaks(arrX_filt_seg*-1, prominence = 50)[0]
        if peaks_filt.shape[0] == 0 or troughs_filt.shape[0] == 0: # no steps detected (there should always be locomotion, but some fragments are just too short to have steps)
            print("No peaks and troughs detected!")    
            continue
        peaks_filt, troughs_filt = utils_processing.are_arrays_alternating(peaks_filt, troughs_filt)
        if type(peaks_filt) != bool :
            peaks_true = []; peak_ids_to_remove= []
            troughs_true = []
            [peaks_true.append(np.argmax(arr_seg[(pf-13):(pf+13)])+pf-13) for pf in peaks_filt if (pf >= 13 and pf+13 < len(arrayX[fon:foff, limb_id_dict[limb]]))]
            peaks_true = np.asarray(peaks_true)
            for i_pt, pt in enumerate(troughs_filt):
                if (pt >= 13 and pt+13 < len(arrayX[fon:foff, limb_id_dict[limb]])):
                    if np.argmin(arr_seg[(pt-13):(pt+13)])+pt-13 not in troughs_true:
                        troughs_true.append(np.argmin(arr_seg[(pt-13):(pt+13)])+pt-13)
                    else:
                        print("AVOIDING DUPLICATE TROUGHS...")
                        peak_ids_to_remove.append(i_pt-1)
                        peaks_true[i_pt] = int(np.mean((peaks_true[i_pt-1], peaks_true[i_pt])))
                elif pt >= 13:
                    troughs_true.append(np.argmin(arr_seg[(pt-13):(len(arrayX[fon:foff, 0])-1)])+pt-13)
                else:
                    troughs_true.append(np.argmin(arr_seg[:(pt+13)])) 

            if len(peaks_true)>0:
                peaks_true = peaks_true[~np.isin(peaks_true, peaks_true[peak_ids_to_remove])]
            troughs_true =np.asarray(troughs_true)
        else:
            print(f"Could not get peaks from {'_'.join(arrayIDer[fon,:])}!")
            continue
       
        # iterate over strides
        if troughs_true.shape[0] > 1 and peaks_true.shape[0] > 1:
            for i in range(troughs_true.shape[0]-1):
                stepStart = troughs_true[i]
                stepEnd = troughs_true[i+1]
                
                dlc_limb = arr_seg[stepStart:stepEnd]
                n = dlc_limb.shape[0]
                delays = np.linspace(-0.5*n, 0.5*n, n).astype(int)
                for bp, corr_arr in zip(limb_list_str, limb_list): # iterate over non-reference bps
                    corrs = []
                    for t_delay in delays: # iterate over time delays
                        try:
                            dlc = arrayX[fon:foff, limb_id_dict[bp]][(stepStart+t_delay):(stepEnd+t_delay)]
                            if np.std(dlc) < 15: # IF MOUSE HOLDS THE WALL WITH THIS LIMB
                                corrs.append(np.nan)
                                continue
                            corr, _ = pearsonr(dlc, dlc_limb)
                            corrs.append(corr)
                        except:
                            # print('Out of range!')
                            corrs.append(np.nan)
                    if not np.all(np.isnan(corrs)):
                        corr_arr.append(delays[np.nanargmax(corrs)] / delays.shape[0]) #appends a single value!
                    else:
                        corr_arr.append(np.nan)
                
                # matched speed and snout-body angles
                speed_arr.append(np.nanmean(speed[fon:foff][(stepStart+delays[0]):(stepEnd+delays[-1])]))
                snoutBody_arr.append(np.nanmean(bodyAngles[fon:foff, 0][(stepStart+delays[0]):(stepEnd+delays[-1])]))
                tailBody_arr.append(np.nanmean(bodyAngles[fon:foff, 2][(stepStart+delays[0]):(stepEnd+delays[-1])]))
                bodyHeight_arr.append(np.nanmean(bodyHeights_rel[fon:foff][(stepStart+delays[0]):(stepEnd+delays[-1])]))
                tailbaseHeight_arr.append(np.nanmean(tailbaseHeights_rel[fon:foff][(stepStart+delays[0]):(stepEnd+delays[-1])]))
                if len(bodyAngles[fon:foff, 0][(stepStart+delays[0]):(stepEnd+delays[-1])]) > 0:
                    bodyAngleRange_arr.append(np.nanmax(bodyAngles[fon:foff,0][(stepStart+delays[0]):(stepEnd+delays[-1])]) - np.nanmin(bodyAngles[fon:foff,0][(stepStart+delays[0]):(stepEnd+delays[-1])]))
                else:
                    bodyAngleRange_arr.append(np.nan)
                
                # full coord array
                dlc_limbX = arrX_filt_seg[stepStart:stepEnd]
                dlc_limbX2 = arrX_filt_seg2[stepStart:stepEnd]
                dlc_limbY = arrY_filt_seg[stepStart:stepEnd]
                
                if n <= stride_pt_max:
                    arrayX_strides = np.append(arrayX_strides, np.append(dlc_limbX, np.zeros(stride_pt_max-n) + np.nan).reshape(-1,1), axis = 1)
                    arrayX_strides2 = np.append(arrayX_strides2, np.append(dlc_limbX2, np.zeros(stride_pt_max-n) + np.nan).reshape(-1,1), axis = 1)
                    arrayY_strides = np.append(arrayY_strides, np.append(dlc_limbY, np.zeros(stride_pt_max-n) + np.nan).reshape(-1,1), axis = 1)
                else:
                    arrayX_strides = np.append(arrayX_strides, (np.zeros(stride_pt_max) + np.nan).reshape(-1,1), axis = 1)
                    arrayX_strides2 = np.append(arrayX_strides2, (np.zeros(stride_pt_max) + np.nan).reshape(-1,1), axis = 1)
                    arrayY_strides = np.append(arrayY_strides, (np.zeros(stride_pt_max) + np.nan).reshape(-1,1), axis = 1)
                    
            stride_freqs = 1/(np.diff(troughs_true)/Config.passiveOpto_config['fps']) #the number of data points from swing-on to swing-on
            if troughs_true.shape[0] > peaks_true.shape[0]: #stride_freqs as long as stride_lengths
                stride_lengths = (arr_seg[peaks_true]-arr_seg[troughs_true[:-1]])/Config.passiveOpto_config["px_per_cm"][limb]
            elif troughs_true.shape[0] == peaks_true.shape[0]: #stride_freqs as long as stride_lengths
                stride_freqs = np.append(stride_freqs, np.nan)
                speed_arr.append(np.nan)
                snoutBody_arr.append(np.nan)
                tailBody_arr.append(np.nan)
                bodyAngleRange_arr.append(np.nan)
                bodyHeight_arr.append(np.nan)
                tailbaseHeight_arr.append(np.nan)
                arrayX_strides = np.append(arrayX_strides, (np.zeros(stride_pt_max) + np.nan).reshape(-1,1), axis = 1)
                arrayX_strides2 = np.append(arrayX_strides2, (np.zeros(stride_pt_max) + np.nan).reshape(-1,1), axis = 1)
                arrayY_strides = np.append(arrayY_strides, (np.zeros(stride_pt_max) + np.nan).reshape(-1,1), axis = 1)
                
                for corr_arr in limb_list:
                    corr_arr.append(np.nan)
                stride_lengths = (arr_seg[peaks_true]-arrayX[fon:foff, limb_id_dict[limb]][troughs_true])/Config.passiveOpto_config["px_per_cm"][limb]
            else:
                print(f"More peaks than troughs! Peaks: {len(peaks_true)} Troughs: {len(troughs_true)}")
                # continue
        else:
            print('Peaks and troughs do not have a length >1 ! Discarding...')
            continue
        
        strideLength = np.append(strideLength, stride_lengths)
        strideFreq = np.append(strideFreq, stride_freqs)
        mouseID = np.append(mouseID, np.repeat(arrayIDer[fon,0], stride_lengths.shape[0]))
        expDate = np.append(expDate, np.repeat(arrayIDer[fon,1], stride_lengths.shape[0]))
        stimFreq = np.append(stimFreq, np.repeat(arrayIDer[fon,2], stride_lengths.shape[0]))
        headLVL = np.append(headLVL, np.repeat(arrayIDer[fon,3], stride_lengths.shape[0]))
        strideNum = np.append(strideNum, np.arange(stride_lengths.shape[0])+1) 
        
        if os.path.exists(Path(outputDir)/"metadataPyRAT_processed.csv"):
            weight, age = utils_processing.get_weight_age(metadata_df, arrayIDer[fon,0], arrayIDer[fon,1])
            weights = np.append(weights, np.repeat(weight, stride_lengths.shape[0]))
            ages = np.append(ages, np.repeat(age, stride_lengths.shape[0]))
            if np.asarray(metadata_df[metadata_df['mouseID']['mouseID']==arrayIDer[fon,0]]['Sex']['Sex'])[0] == 'f':
                sexes = np.append(sexes, np.repeat(1, stride_lengths.shape[0])) # append 1 if FEMALE
            else:
                sexes = np.append(sexes, np.repeat(0, stride_lengths.shape[0])) # append 0 if MALE
            
            if 'rl' in arrayIDer[0,3]:
                headHW = np.append(headHW, np.repeat(headHeight_scaled[fon]/weight, stride_lengths.shape[0]))
    
    # COMPUTE MEAN LIMB PHASES + CORRECT FOR INCONSISTENT TRACKING & WALL-HOLDING WITH NANs
    for mean_limb_arr, corr_limb_arr in zip(mean_limb_list,[limb_list[1:3], limb_list[3:5], limb_list[5:7]]) :
        diff = abs(np.asarray(corr_limb_arr[0])-np.asarray(corr_limb_arr[1])) # difference between the two markers
        mean_corr = np.mean(corr_limb_arr, axis = 0) # NOT nanmean: if one is nan, the mean should be nan (discard that stride!)
        mean_corr[diff>0.05] = np.nan # if difference between the two markers big, at least one is poorly tracked
        [mean_limb_arr.append(m) for m in mean_corr]
        
        # run to here
        
    yremv = np.where(np.nanmin(arrayY_strides, axis = 0) < 200)[0]
    arrayY_strides[:, yremv] = np.zeros((arrayY_strides.shape[0], len(yremv))) + np.nan
    arrayX_strides[:, yremv] = np.zeros((arrayY_strides.shape[0], len(yremv))) + np.nan
    arrayX_strides2[:, yremv] = np.zeros((arrayY_strides.shape[0], len(yremv))) + np.nan
    
    diffX = np.diff(arrayX_strides, axis = 0)
    diffY = np.diff(arrayY_strides, axis = 0)
    dists = np.nansum(np.sqrt(diffX**2 + diffY**2), axis = 0) # nan columns become 0
    dists[dists == 0] =np.nan
    
    if os.path.exists(Path(outputDir)/"metadataPyRAT_processed.csv") and 'rl' in arrayIDer[0,3]:
        df = pd.DataFrame(np.vstack((mouseID, expDate, stimFreq, headLVL, weights, ages, headHW, sexes,
                                     strideNum, strideFreq, strideLength, limb_list[0], limb_list[1],
                                     limb_list[2], limb_list[3], limb_list[4], limb_list[5], 
                                     limb_list[6], mean_limb_list[0], mean_limb_list[1], mean_limb_list[2],
                                     speed_arr, snoutBody_arr, tailBody_arr, dists, bodyHeight_arr, 
                                     bodyAngleRange_arr, tailbaseHeight_arr)).T,
                          columns = ['mouseID', 'expDate', 'stimFreq', 'headLVL', 'weight','age', 'headHW',
                                     'sex','strideNum', 'strideFreq', 'strideLength',  limb_list_str[0], 
                                     limb_list_str[1], limb_list_str[2], limb_list_str[3], 
                                     limb_list_str[4], limb_list_str[5], limb_list_str[6], mean_limb_str[0],
                                     mean_limb_str[1], mean_limb_str[2], 'speed', 'snoutBodyAngle', 'tailBodyAngle', 
                                     'limbDist', 'bodyHeight_rel', 'bodyAngleRange', 'tailbaseHeight_rel'])
    else:
        df = pd.DataFrame(np.vstack((mouseID, expDate, stimFreq, headLVL, weights, ages, sexes,strideNum, 
                                     strideFreq, strideLength, limb_list[0], limb_list[1], 
                                     limb_list[2], limb_list[3], limb_list[4], limb_list[5], 
                                     limb_list[6], mean_limb_list[0], mean_limb_list[1], mean_limb_list[2],
                                     speed_arr, snoutBody_arr, tailBody_arr, dists, bodyHeight_arr, 
                                     bodyAngleRange_arr, tailbaseHeight_arr)).T,
                          columns = ['mouseID', 'expDate', 'stimFreq', 'headLVL', 'weight','age', 
                                     'sex','strideNum', 'strideFreq', 'strideLength',  limb_list_str[0], 
                                     limb_list_str[1], limb_list_str[2], limb_list_str[3], 
                                     limb_list_str[4], limb_list_str[5], limb_list_str[6], mean_limb_str[0],
                                     mean_limb_str[1], mean_limb_str[2], 'speed', 'snoutBodyAngle', 'tailBodyAngle',
                                     'limbDist', 'bodyHeight_rel', 'bodyAngleRange', 'tailbaseHeight_rel'])
    df.to_csv(os.path.join(outputDir, yyyymmdd + f'_strideParams{appdx}_{limb}.csv'))
    
    np.save(Path(outputDir)/ f"{yyyymmdd}_limbX_strides{appdx}_{limb}.npy", arrayX_strides2) 
    np.save(Path(outputDir)/ f"{yyyymmdd}_limbY_strides{appdx}_{limb}.npy", arrayY_strides) 
 
def get_foot_placement_array(param = 'levels', 
                             data = 'preOpto', 
                             outputDir = Config.paths["passiveOpto_output_folder"], 
                             appdx = ""):
    """
    stacks limb xy position data during locomotion or before optostim
    
    PARAMETERS
    ----------
    param (str): was important for grouping plots in an extended version of
                this function ('snoutBodyAngle', 'headHW', 'levels')
    data (str): dataset to use ('preOpto' or 'locom')
    outputDir (str, optional) : path to output folder
    appdx (str) : "" (empty) or "incline" for head height and incline trials respectively
    
    WRITES
    ----------
    f'{yyyymmdd}_limbPositionRegressionArray_{data}_{param}{appdx}.csv'
    """
    if appdx != "":
        appdx = f"_{appdx}"
    
    limbs = ['lH1', 'lH2', 'rH1', 'rH2', 'lF1', 'lF2', 'rF1', 'rF2'] # limbX and limbY have this order too
    
    # LOAD THE DATA
    if data == 'preOpto':
        df, yyyymmdd = data_loader.load_processed_data(outputDir, dataToLoad = 'passiveOptoData', appdx = appdx)
        preOpto = df.loc[:200, (slice(None), slice(None), slice(None), slice(None), limbs)]
        
        limbX = np.empty((preOpto.shape[0]*int(preOpto.shape[1]/16), 8)) # 8 limbs x 2 coords
        limbY = np.empty((preOpto.shape[0]*int(preOpto.shape[1]/16), 8)) # 8 limbs x 2 coords
        arrIDer = np.repeat(np.asarray(list(preOpto.loc[:, (slice(None), slice(None), slice(None), slice(None), 'lH1', 'y')].columns))[:,:4], preOpto.shape[0], axis = 0)
        for i, limb in enumerate(limbs):
            limbX[:,i] = np.concatenate(np.vstack((np.asarray(preOpto.loc[:, (slice(None), slice(None), slice(None), slice(None), limb, 'x')]))).T)
            limbY[:,i] = np.concatenate(np.vstack((np.asarray(preOpto.loc[:, (slice(None), slice(None), slice(None), slice(None), limb, 'y')]))).T)
        
    elif data == 'locom':            
        limbX, yyyymmdd = data_loader.load_processed_data(outputDir, dataToLoad = 'limbX', appdx = appdx)
        limbY, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'limbY', appdx = appdx, yyyymmdd = yyyymmdd)
        arrIDer, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'arrayIdentifier', appdx = appdx, yyyymmdd = yyyymmdd)       
        
    if param == 'snoutBodyAngle':
        bodyAngles, _ = data_loader.load_processed_data(outputDir, 
                                                        dataToLoad = 'bodyAngles', 
                                                        yyyymmdd = yyyymmdd, 
                                                        appdx = appdx)
        bodyAngles_sub = bodyAngles.loc[:, (slice(None), slice(None), slice(None), slice(None), 'snoutBody')]
        multiindex_cols = bodyAngles_sub.columns.droplevel(-1)
        
        arrIDer_df = pd.DataFrame(arrIDer, columns=['mouseID', 'expDate', 'stimFreq', 'headLVL'])
        bodyAngles_col_df = pd.DataFrame(bodyAngles_sub.columns.to_flat_index().map(lambda x: x[:-1]).tolist(),
                                         columns=['mouseID', 'expDate', 'stimFreq', 'headLVL'])
        SBA_means = bodyAngles_sub.mean(axis=0)
        
        SBA_means = pd.DataFrame({
            'mean': SBA_means.values,
            'mouseID': bodyAngles_col_df['mouseID'],
            'expDate': bodyAngles_col_df['expDate'],
            'stimFreq': bodyAngles_col_df['stimFreq'],
            'headLVL': bodyAngles_col_df['headLVL'],
            })
        
        merged_DF = arrIDer_df.merge(SBA_means, on=['mouseID', 'expDate', 'stimFreq', 'headLVL'], how='left')
        arrIDer = merged_DF.drop(['headLVL'], axis=1)
           
    df_reg = pd.DataFrame(np.hstack((arrIDer, limbX, limbY)), columns = np.concatenate((['mouseID'],['expDate'],['stimFreq'], ['trialType'], [x+'x' for x in limbs], [x+'y' for x in limbs])) )   
    df_reg.to_csv(os.path.join(outputDir, yyyymmdd + f'_limbPositionRegressionArray_{data}_{param}{appdx}.csv'))
 
def cot(data, 
        param, 
        dataToPlot, 
        group_num = None, 
        data_split = None,
        wasserstein = False):
    """
    computes circular optimal transport for all pairs of conditions defined by 
    param, e.g. 'headLVL'
    
    PARAMETERS
    ----------
    data (dataframe) : input dataframe ('strideParams') - after filtering by mouseID/category but before making headLVL numerical
    param (str) : strideParams df column name of the parameter to plot (headLVL, stimFreq, snoutBodyAngle, speed)
    dataToPlot (str) : data to plot, e.g. 'speed' or 'rH1'
    group_num (int) : used for splitting data only if data_split not supplied
    data_split (None or list) : a list of group boundaries or None  
    wasserstein (bool) : computes Wassertstein distance instead of COT if true
    
    RETURNS
    ----------
    cot_df (dataframe) : group_num x group_num df of COT values
    trial_num_dict (dictionary) : dictionary with categories as keys as stride numbers as values
    trial_num_arr (numpy array) : group_num x group_num array of per-category stride numbers
    uniform_df (dataframe) : 1 x group_num df of COT values for data vs uniform distribution 
    """
    param_limb = False
    if param == 'stimFreq':
        cond_list = np.unique(data[param]) 
        sorted_ids = np.argsort([int(c[:-2]) for c in cond_list]) 
        cond_list = cond_list[sorted_ids]
        cond_titles = cond_list
    else:
        if param == 'headLVL' or param == 'trialType':
            if 'rl' in data[param].iloc[0]:
                data[param] = [(int(d[2:])*-1)+6 for d in data[param]]
            elif 'deg' in data[param].iloc[0]:
                data[param] = [int(d[3:])*-1 for d in data[param]]
            data_relevant = data[param]
        
        elif param == 'snoutBodyAngle' or param == 'speed':
            data_relevant = utils_processing.remove_outliers(data[param])
        
        elif 'lF' in param or 'rF' in param or 'lH' in param or 'rH' in param:
            data_relevant = data[param]
            param_limb = True
        
        else:
            raise ValueError("Unrecognisable param supplied!")
            
        if type(data_split) != list:
            if group_num == None:
                group_num = int(input("Enter the desired number of groups: "))    
            data_split = np.linspace(data_relevant.min(), data_relevant.max(), group_num+1)
       
        data_grouped = data.groupby(pd.cut(data[param], data_split)) 
        group_row_ids = data_grouped.groups
        cond_list = group_row_ids  
    
    import itertools
    import scipy.stats
    ids = [p for p in itertools.product(np.arange(len(cond_list)), repeat = 2)] [::-1]#finds all pairs of conditions
    cot_arr = np.empty((len(cond_list), len(cond_list)))
    cot_arr[:] = np.nan
    trial_num_arr = np.zeros_like(cot_arr)
    trial_num_arr[:] =  np.nan
    uniform_arr = np.empty((1, len(cond_list)))
    uniform_arr[:] = np.nan
     
    for i in ids:      
        # initialise arrays for storing histogram values (not necessarily normalised)
        p = []; q = []
        trial_num_comparison = []
        
        # wrangle data to populate the above arrays
        for k, probdist in enumerate([p,q]):
            if param == 'stimFreq':
                data_sub = data[data[param] == cond_list[i[k]]][dataToPlot]
            elif param == 'snoutBodyAngle' or param == 'speed' or param == 'headLVL' or param_limb or param == 'trialType':
                grouped_dict = {key:data.loc[val,[dataToPlot]].values for key,val in group_row_ids.items()} 
                keys = list(grouped_dict.keys())
                cond_list = keys
                if not wasserstein:
                    cond_titles = [f"({k.left:.0f}, {k.right:.0f}]" for k in list(keys)]
                else:
                    cond_titles = [f"({k.left:.1f}, {k.right:.1f}]" for k in list(keys)]
                data_sub = grouped_dict[keys[i[k]]]
            
            data_sub = data_sub[~np.isnan(data_sub)]
            
            trial_num_comparison.append(data_sub.shape[0])
            if dataToPlot in Config.passiveOpto_config["px_per_cm"].keys() or wasserstein:         
                if data_sub.shape[0]>Config.passiveOpto_config['stride_num_threshold']:
                    [probdist.append(x) for x in np.asarray(data_sub)]
                    probdist = np.asarray(probdist)
                else:
                    [probdist.append(x) for x in np.repeat(np.nan, data_sub.shape[0])]
                
            else:
                raise ValueError("Cannot apply circular OT to non-circular data. Implement Wasserstein distance!")
      
            if i[0] == i[1]: # main diagonal
                trial_num_arr[i[0], i[1]] = data_sub.shape[0]  
        
        if i[0] > i[1]:
            trial_num_arr[i[0],i[1]] = np.min(trial_num_comparison)
            # print(i, trial_num_comparison)
            
        # compute (or retrieve) COT
        if np.isnan(cot_arr[i[1], i[0]]): # the symmetrical value has not been computed
            if len(p) > Config.passiveOpto_config['stride_num_threshold'] and len(q) > Config.passiveOpto_config['stride_num_threshold']:
                # print(i, len(p), len(q))
                if not wasserstein:
                    cot = utils_math.circular_optimal_transport(p, q, dataType = 'unit') # cot
                else:
                    cot = scipy.stats.wasserstein_distance(p,q) # should not have linear data
            else:
                cot = np.nan
        else:
            cot = cot_arr[i[1], i[0]]
        
        if np.isnan(uniform_arr[0,i[0]]):
            if len(p) > Config.passiveOpto_config['stride_num_threshold']:
                if not wasserstein:
                    uniform = np.random.uniform(-0.5,0.5, len(p))
                    cot_uni = utils_math.circular_optimal_transport(p, uniform, dataType = 'unit')
                else:
                    uniform = np.random.uniform(np.min(p),np.max(p), len(p))
                    cot_uni = scipy.stats.wasserstein_distance(p, uniform)
            else:
                cot_uni = np.nan
            uniform_arr[0,i[0]] = cot_uni
        
        # store COT in half of jsd_arr
        if i[1] < i[0]: # COTs are symmetrical so this would be redundant
            cot_arr[i[0], i[1]] = cot
                
    
    cot_df = pd.DataFrame(cot_arr, columns = cond_titles, index = cond_titles)
    uniform_df = pd.DataFrame(uniform_arr, columns = cond_titles)
    
    trial_num_dict = {}
    for i in ids[::(int(np.sqrt(len(ids))+1))]: # (0,0) (1,1) etc
        trial_num_dict[cond_list[i[0]]] = trial_num_arr[i[0], i[1]]
    
    return cot_df, trial_num_dict, trial_num_arr, uniform_df

def compute_locomotor_params(outputDir = Config.paths["passiveOpto_output_folder"], 
                             appdx = ""
                             ):  
    """
    computes per-trial locom latency, mean belt speed (a.u.), distance (AUC; a.u.), duration?, 
    probability across trials <-- in another function because this should be grouped in various ways (based on freq and head lvl)
    
    beltSpeedData contains belt speed measurement between 200fps before optoON and 200fps after optoOFF
    """
    if appdx != "":
        appdx = f"_{appdx}"
        
    beltSpeedData, yyyymmdd = data_loader.load_processed_data(outputDir, 
                                                              dataToLoad = 'beltSpeedData', 
                                                              appdx = appdx)
    beltSpeedData[beltSpeedData < 0] = 0
    beltSpeedData.sort_index(axis = 1, inplace = True)
    fps = Config.passiveOpto_config['fps'] 
    on_id = int(fps*0.5)
    locom_threshold = 0.01 * 150/3.3 #0.01 was used before recalibrating arduino output; 
                                     #the old calibration values are used for convertion to speed
    
    # ANGLES
    bodyAngles, _ = data_loader.load_processed_data(outputDir, 
                                                    dataToLoad = 'bodyAngles', 
                                                    yyyymmdd = yyyymmdd, 
                                                    appdx = appdx)
    anglesPreOpto = bodyAngles.loc[:200, (slice(None), slice(None), slice(None), slice(None), 'snoutBody')].mean(axis=0)
    anglesOptoON = bodyAngles.loc[200:2199, (slice(None), slice(None), slice(None), slice(None), 'snoutBody')]
    anglesOptoONmean = bodyAngles.loc[200:2199, (slice(None), slice(None), slice(None), slice(None), 'snoutBody')].mean(axis=0)
    anglesOptoONmean.sort_index()
    anglesLocom = np.nanmean(anglesOptoON[beltSpeedData.iloc[on_id:-on_id,:]>locom_threshold], axis = 0)
    
    if not np.all(anglesOptoONmean.droplevel(-1).index == beltSpeedData.columns):
        print("Data not sorted properly!")
    
    mouseID = beltSpeedData.columns.get_level_values(0)
    expDate = beltSpeedData.columns.get_level_values(1)
    stimFreq = beltSpeedData.columns.get_level_values(2)
    headLVL = beltSpeedData.columns.get_level_values(3)
        
    is_locomoting = (beltSpeedData.iloc[on_id:-on_id,:]>locom_threshold).astype(int)
    
    #DURATION as a fraction of optoON
    duration = np.sum(is_locomoting, axis=0)/is_locomoting.shape[0] 
    
    #LATENCY in seconds
    # nan if speed > 0.01 at any point during the 150 frames preceding the stim
    # nan if speed never > 0.01 during the stim
    latency = np.asarray([np.where(is_locomoting.iloc[:,col] == 1)[0][0]/fps if (len(np.where(is_locomoting.iloc[:,col] == 1)[0]) > 0 and np.all(beltSpeedData.iloc[50:on_id, col]<locom_threshold)) else np.nan for col in range(is_locomoting.shape[1])])
    
    #DISTANCE (AUC in a.u.)
    distance = np.sum(beltSpeedData.iloc[on_id:-on_id,:], axis =0)
    
    #MEAN SPEED
    meanSpeed = np.nanmean(beltSpeedData[beltSpeedData.iloc[on_id:-on_id,:]>locom_threshold], axis =0)
    
    #MAX SPEED
    maxSpeed = np.max(beltSpeedData[beltSpeedData.iloc[on_id:-on_id,:]>locom_threshold], axis =0)
    
    #MEDIAN SPEED
    medianSpeed = np.nanmedian(beltSpeedData[beltSpeedData.iloc[on_id:-on_id,:]>locom_threshold], axis =0)
    
    #LOCOMOTION YES? for PROBABILITY --> the actual probability will have to be computed for various conditions (stimFreq, headLVL, mouse) separately
    locom_yes = np.asarray([np.any(is_locomoting.iloc[:,col]==1) if np.all(beltSpeedData.iloc[50:on_id, col]<locom_threshold) else np.nan for col in range(is_locomoting.shape[1])])
    
    if os.path.exists(Path(outputDir)/"metadataPyRAT_processed.csv") and 'rl' in headLVL[0]:
        metadata_df = data_loader.load_processed_data(outputDir, 
                                                      dataToLoad = 'metadataProcessed', 
                                                      yyyymmdd = yyyymmdd, 
                                                      appdx = appdx)
        headHeight_scaled = [(-int(x[2:])+24.5)/Config.passiveOpto_config["mm_per_g"] for x in headLVL]   
        # x/22 = (x+5)/26, so x = 27.5 = rl-3
        # in this case, x/22 = 1.25 mm/g, so for the *just*-reachable height to be 1, I divide by 1.25
        # this quantity is ready to be divided by weight
        
        headHW = []
        for m, e, h in zip(mouseID, expDate, headHeight_scaled):
            weight, _ = utils_processing.get_weight_age(metadata_df, m, e)
            headHW = np.append(headHW, h/weight)       
    
        df = pd.DataFrame(np.vstack((mouseID, expDate, stimFreq, headLVL, headHW, anglesLocom, anglesPreOpto,duration, latency, distance, meanSpeed, medianSpeed, maxSpeed, locom_yes)).T,
                      columns = ['mouseID', 'expDate', 'stimFreq', 'headLVL', 'headHW', 'snoutBodyAngle', 'snoutBodyAngle_preOpto', 'duration', 'latency', 'distance', 'meanSpeed', 'medianSpeed', 'maxSpeed', 'locomYES'])
    else:
        df = pd.DataFrame(np.vstack((mouseID, expDate, stimFreq, headLVL, anglesLocom, anglesPreOpto, duration, latency, distance, meanSpeed,medianSpeed, maxSpeed, locom_yes)).T,
                      columns = ['mouseID', 'expDate', 'stimFreq', 'headLVL', 'snoutBodyAngle', 'snoutBodyAngle_preOpto', 'duration', 'latency', 'distance', 'meanSpeed', 'medianSpeed', 'maxSpeed','locomYES'])
    
    df.to_csv(os.path.join(outputDir, yyyymmdd + f'_locomParams{appdx}.csv'))    
        
def cot_wrapper(param,
                dataToPlot,
                data_split = None,  
                repetitions = 10000, 
                group_num = 5,
                outputDir = Config.paths["passiveOpto_output_folder"],
                appdx = "",
                wasserstein = False,
                phaseshift = False):
    """
    computes circular optimal transport for all pairs of conditions defined by 
    param and performs a permutation test
    
    PARAMETERS
    ----------
    param (str) : strideParams df column name of the parameter to plot (headLVL, stimFreq, snoutBodyAngle, speed)
    dataToPlot (list of strs) : data to plot from strideParams.columns, e.g. 'speed' or 'rH0'
    data_split (None or list) : a list of group boundaries or None   
    repetitions (int) : number of data shufflings
    group_num (int) : used for splitting data only if data_split not supplied
    outputDir (str, optional) : path to output folder
    appdx (str) : "" (empty) or "incline" for head height and incline trials respectively
    wasserstein (bool) : computes Wassertstein distance instead of COT if true
    phaseshift (bool) : adds 1 to all phases below 0 if True (useful for lF0 Wasserstein = True plots)
    
    WRITES
    ----------
    f'{yyyymmdd}_COTpvals_{limb}_{param}{appdx}.csv' : p values from pairwise permutation tests 
                                                       across mice between groups
                                                       [group_num x group_num dataframe]
    f'{yyyymmdd}_COT_{limb}_{param}{appdx}.csv')) : mean pairwise COT across mice 
                                                    between groups [group_num x group_num dataframe]
    f'{yyyymmdd}_COTptext_{limb}_{param}{appdx}.csv')) : stars associated with the 
                                                         permutation test between groups
                                                         [group_num x group_num dataframe]
    f'{yyyymmdd}_COT_UNIFORMpvals_{limb}_{param}{appdx}.csv')) : p values from permutation tests
                                                                 between data and uniform distrib
                                                                 across mice
                                                                 [1 x group_num dataframe]
    f'{yyyymmdd}_COT_UNIFORM_{limb}_{param}{appdx}.csv')) : mean COT between each data group
                                                            and uniform distribution across
                                                            mice [1 x group_num dataframe]
    f'{yyyymmdd}_COT_UNIFORMptext_{limb}_{param}{appdx}.csv')) : stars associated with the 
                                                                 permutation test between
                                                                 data groups and uniform distrib
                                                                 [1 x group_num dataframe]
    
    EXAMPLES
    ----------
    FOR MOTORISED TREADMILL INCLINE
    cot_wrapper(param = 'trialType',
                    dataToPlot = ['lF0'],
                    data_split = [-40.0000001,-24,-8,8,24,40],  
                    repetitions = 10, 
                    group_num = 5,
                    outputDir = Config.paths["mtTreadmill_output_folder"],
                    appdx = "",
                    wasserstein = False,
                    phaseshift = False)
    
    FOR MOTORISED TREADMILL LEVEL
    cot_wrapper(param = 'lF0',
                dataToPlot = ['snoutBodyAngle'],
                data_split = [-0.000000001,0.2,0.4,0.6,0.8],  
                repetitions = 10000, 
                group_num = 4,
                outputDir = Config.paths["mtTreadmill_output_folder"],
                appdx = "",
                wasserstein = True,
                phaseshift = True)
    
    FOR PASSIVE TREADMILL INCLINE
    cot_wrapper(param = 'headLVL',
                dataToPlot = ['lF0'],
                data_split = [-40.0000001,-24,-8,8,24,40],  
                repetitions = 10000, 
                group_num = 5,
                outputDir = Config.paths["passiveOpto_output_folder"],
                appdx = "incline",
                wasserstein = False,
                phaseshift = False)
    
    FOR PASSIVE TREADMILL SNOUT-BODY-ANGLE
    cot_wrapper(param = 'snoutBodyAngle',
                dataToPlot = ['lF0'],
                data_split = [141,147,154,161,167,174],  
                repetitions = 10000, 
                group_num = 5,
                outputDir = Config.paths["passiveOpto_output_folder"],
                appdx = "",
                wasserstein = False,
                phaseshift = False)
    """
    if appdx != "":
        appdx = f"_{appdx}"
    
    if wasserstein:
        wasserstein_bool = True
        stat_str = 'WASSERSTEIN'
    else:
        wasserstein_bool = False
        stat_str = 'COT'
        
    df, yyyymmdd, limbREF = data_loader.load_processed_data(outputDir = outputDir, 
                                                            dataToLoad = "strideParams", 
                                                            appdx = appdx)
    
    if phaseshift:
        df[param] = [x+1 if x<0 else x for x in df[param]]
    
    if outputDir == Config.paths["passiveOpto_output_folder"]:
        cfg = Config.passiveOpto_config
        mouse_str = 'mice'
    elif outputDir == Config.paths["mtTreadmill_output_folder"]:
        cfg = Config.mtTreadmill_config
        if '2021' in yyyymmdd:
            mouse_str = 'mice_level'
        else:
            mouse_str = 'mice_incline'
    else:
        raise ValueError("OutputDir failed to specify a config dict!")
    
    # subset mice
    for m in np.setdiff1d(np.unique(df['mouseID']), cfg[mouse_str]):
        df = df[df['mouseID'] != m]
    mice = np.unique(df['mouseID'])
    
    # initialise arrays
    cot_arr_4d = np.empty((mice.shape[0], group_num, group_num, len(dataToPlot)))
    cot_arr_4d[:] = np.nan
    cot_arr_5d_perm = np.empty((mice.shape[0], repetitions, group_num, group_num, len(dataToPlot)))
    cot_arr_5d_perm[:] = np.nan
    
    uniform_arr_3d = np.empty((mice.shape[0], group_num, len(dataToPlot)))  
    uniform_arr_3d[:] = np.nan
    uniform_arr_4d_perm = np.empty((mice.shape[0], repetitions, group_num, len(dataToPlot))) 
    uniform_arr_4d_perm[:] = np.nan
    
    for im, m in enumerate(mice):
        print(f'Processing mouse {m}...')
        for il, limb in enumerate(dataToPlot):
            cot_arr, _, trial_num_arr, cot_uniform = cot(data = df[df['mouseID'] == m], 
                                                         param = param,
                                                         dataToPlot = limb, 
                                                         data_split = data_split, 
                                                         group_num = group_num,
                                                         wasserstein = wasserstein_bool)
            print(m, trial_num_arr) 
            print(m, cot_arr)                                
            cot_arr[trial_num_arr < cfg['stride_num_threshold']] = np.nan
            cot_arr_4d[im, :, :, il] = cot_arr
            uniform_arr_3d[im, :, il] = cot_uniform # compare each data group with a uniform distribution
            
    for rep in range(repetitions):
        print(f'Running permutation #{rep}')
        for im, m in enumerate(mice):
            df_sub = df[df['mouseID'] == m]
            for il, limb in enumerate(dataToPlot):
                pd.options.mode.chained_assignment = None
                df_sub[limb] = np.random.permutation(df_sub[limb].values) # shuffles one column relative to others
                df_sub_c = copy.deepcopy(df_sub)
                cot_arr, _, trial_num_arr, cot_uniform = cot(data = df_sub_c, 
                                                             param = param,
                                                             dataToPlot = limb, 
                                                             data_split = data_split, 
                                                             group_num = group_num,
                                                             wasserstein = wasserstein_bool)
                cot_arr[trial_num_arr < cfg['stride_num_threshold']] = np.nan
                cot_arr_5d_perm[im, rep, :, :, il] = cot_arr
                uniform_arr_4d_perm[im, rep, :, il] = cot_uniform
    
    ids = [p for p in itertools.product(np.arange(cot_arr.shape[1]), repeat = 2)] [::-1] # finds all pairs of conditions
    ids_c = [p for p in itertools.combinations(np.arange(cot_arr.shape[1]), 2)] 
    
    pval_arr_3d = np.empty((group_num, group_num, len(dataToPlot)))
    pval_arr_3d[:] = np.nan
    pval_uni_arr_2d = np.empty((group_num, len(dataToPlot)))
    pval_uni_arr_2d[:] = np.nan
    
    cot_dict = {}
    cot_perm_dict = {}
    cot_uni_dict = {}
    cot_uni_perm_dict= {}
    pval_text_dict = {}
    pval_text_uni_dict = {}
    pval_cot_thresh = np.asarray(FigConfig.p_thresholds) / len(ids_c) # Bonferroni corrected
    pval_cot_perm_thresh =  np.asarray(FigConfig.p_thresholds) / group_num # Bonferroni corrected
    for il, limb in enumerate(dataToPlot):
        cot_df_mean = np.nanmean(cot_arr_4d[:,:,:,il], axis = 0) # mean COT across mice
        cot_df_perm_mean = np.nanmean(cot_arr_5d_perm[:,:,:,:,il], axis = 0) # mean COT across mice for each permutation      
        cot_dict[limb] = pd.DataFrame(cot_df_mean, columns = cot_arr.columns, index = cot_arr.columns)
        cot_perm_dict[limb] = pd.DataFrame(np.nanmean(cot_df_perm_mean, axis = 0), columns = cot_arr.columns, index = cot_arr.columns)
        
        # deal with uniform distributions
        cot_uni_df_mean =  np.nanmean(uniform_arr_3d[:,:,il], axis = 0) # mean COT across mice
        cot_uni_dict[limb] = pd.DataFrame(cot_uni_df_mean, index = cot_arr.columns)
        cot_uni_df_perm_mean = np.nanmean(uniform_arr_4d_perm[:,:,:,il], axis = 0) # mean permuted COT across  mice
        cot_uni_perm_dict[limb] = pd.DataFrame(np.nanmean(cot_uni_df_perm_mean, axis = 0), index = cot_arr.columns)
       
        for ix in ids:
            print(ix)
            if limb not in pval_text_dict.keys():
                pval_text_dict[limb] = []
                pval_text_uni_dict[limb] = []
            if ix[0]>ix[1]:
                # COT
                critical_cot = cot_df_mean[ix[0],ix[1]]
                shuffled_cots = cot_df_perm_mean[:, ix[0], ix[1]]
                upper_frac =  (shuffled_cots>critical_cot).sum()
                
                if upper_frac > repetitions/2:
                    p = 1 - (upper_frac/repetitions)
                else:
                    p = upper_frac/repetitions
                pval_arr_3d[ix[0],ix[1],il] = p
                
                # add stats stars to pval_text dict
                star_num = (p<pval_cot_thresh).sum()
                pval_text_dict[limb].append('*'*star_num)
                
                # deal with uniforms
                if np.isnan(pval_uni_arr_2d[ix[0], il]) or np.isnan(pval_uni_arr_2d[ix[1], il]):
                    if np.isnan(pval_uni_arr_2d[ix[0], il]):
                        ix_col = ix[0]
                    elif np.isnan(pval_uni_arr_2d[ix[1], il]):
                        ix_col = ix[1]
                    uniform_cots = cot_uni_df_perm_mean[:, ix_col]
                    upper_frac_uni = (uniform_cots>critical_cot).sum()
                    if upper_frac_uni > repetitions/2:
                        p = 1 - (upper_frac_uni/repetitions)
                    else:
                        p = upper_frac_uni/repetitions
                    pval_uni_arr_2d[ix_col,il] = p
                    
                    # add stats stars
                    star_num = (p<pval_cot_perm_thresh).sum()
                    pval_text_uni_dict[limb].append('*'*star_num)
            
            else:
                pval_text_dict[limb].append('')  
          
                
        pval_df_limb = pd.DataFrame(pval_arr_3d[:,:,il], columns = cot_arr.columns, index = cot_arr.columns)
        pval_df_uniform = pd.DataFrame(pval_uni_arr_2d[:,il], index = cot_arr.columns)
        pval_text_dict[limb] = np.flip(np.asarray(pval_text_dict[limb]).reshape(group_num,group_num)) # flip because ids start with (4,4)
        pval_text_df = pd.DataFrame(pval_text_dict[limb], index = cot_arr.columns) 
        pval_text_uni_df = pd.DataFrame(pval_text_uni_dict[limb], index = cot_arr.columns)
        
        pval_df_limb.to_csv(os.path.join(outputDir, yyyymmdd + f'_{stat_str}pvals_{limb}_{param}{appdx}.csv'))
        cot_dict[limb].to_csv(os.path.join(outputDir, yyyymmdd + f'_{stat_str}_{limb}_{param}{appdx}.csv'))
        pval_text_df.to_csv(os.path.join(outputDir, yyyymmdd + f'_{stat_str}ptext_{limb}_{param}{appdx}.csv'))
        pval_df_uniform.to_csv(os.path.join(outputDir, yyyymmdd + f'_{stat_str}_UNIFORMpvals_{limb}_{param}{appdx}.csv'))
        cot_uni_dict[limb].to_csv(os.path.join(outputDir, yyyymmdd + f'_{stat_str}_UNIFORM_{limb}_{param}{appdx}.csv'))
        pval_text_uni_df.to_csv(os.path.join(outputDir, yyyymmdd + f'_{stat_str}_UNIFORMptext_{limb}_{param}{appdx}.csv'))
        
def generate_ideal_gaits(outputDir, sd = 0.06, refLimb = 'lH1'):
    """
    generates idealised trot, bound, transverse gallop, rotary gallop
    based on data presented in Bellardita and Kiehn 2015
    and their definitions of strict alternation/synchrony
    """
    np.random.seed(13)
    
    # phase simulations
    inphase = np.random.normal(loc = 0, scale = sd, size = 1000) # in phase
    outofphase_bound = np.random.normal(loc = -0.4, scale = sd, size = 1000) # oo-phase
    outofphase_gallop_R_hmlg = np.random.normal(loc = 0.15, scale = sd, size = 1000) # oo-phase ref leading
    outofphase_gallop_NR_hmlg = np.random.normal(loc = 0.15, scale = sd, size = 1000) # oo-phase non-ref leading
    outofphase_gallop_R_hmlt = np.random.normal(loc = 0.3, scale = sd, size = 1000) # oo-phase
    outofphase_gallop_R_diag = np.random.normal(loc = -0.3, scale = sd, size = 1000) # oo-phase
    antiphase = np.random.normal(loc = -0.5, scale = sd, size = 1000) # anti-phase
    antiphase = np.concatenate((antiphase[antiphase >= -0.5], antiphase[antiphase <= -0.5]+1))
    
    # idealised gaits
    trot = {'homologous': antiphase, 'homolateral': antiphase, 'diagonal': inphase}
    bound = {'homologous': inphase, 'homolateral': outofphase_bound, 'diagonal': outofphase_bound}
    transverse_gallop_R = {'homologous': outofphase_gallop_R_hmlg, 'homolateral': outofphase_gallop_R_hmlt, 'diagonal': outofphase_gallop_R_diag}
    rotary_gallop_R = {'homologous': outofphase_gallop_R_hmlg, 'homolateral': outofphase_gallop_R_diag, 'diagonal': outofphase_gallop_R_hmlt}
    transverse_gallop_NR = {'homologous': outofphase_gallop_NR_hmlg, 'homolateral': outofphase_gallop_R_diag, 'diagonal': outofphase_gallop_R_hmlt}
    rotary_gallop_NR = {'homologous': outofphase_gallop_NR_hmlg, 'homolateral': outofphase_gallop_R_hmlt, 'diagonal': outofphase_gallop_R_diag}

    # reference limb dictionary
    refLimbDict = {'lH1': {'lF0': 'homolateral', 'rF0': 'diagonal', 'rH0': 'homologous'}, 
                   'rH1': {'rF0': 'homolateral', 'lF0': 'diagonal', 'lH0': 'homologous'}}
    gaitDict = {'trot': trot, 
                'bound': bound, 
                'transverse_gallop_R': transverse_gallop_R,
                'rotary_gallop_R': rotary_gallop_R, 
                'transverse_gallop_NR': transverse_gallop_NR,
                'rotary_gallop_NR': rotary_gallop_NR}
        
    # gaitDict_ref = {gait: {k: dct[refLimbDict[refLimb][k]] for k in refLimbDict[refLimb].keys()} for gait, dct in gaitDict.items()}
     
    import pickle
    pickle.dump(gaitDict, open(os.path.join(outputDir, 'idealisedGaitDict.pkl'), "wb" ))                                               
    # pickle.dump(gaitDict_ref, open(os.path.join(outputDir, 'idealisedGaitDict_{refLimb}.pkl'), "wb" )) 


def get_limb_amplitudes(param,
                        group_num =5,
                        limbRef = 'lH1', 
                        appdx = "", 
                        outputDir = Config.paths["passiveOpto_output_folder"]):
    """
    computes limb kinematics (x and y coords) per param category
    
    param (str) : 'snoutBodyAngle' or 'headLVL'
    """
    if appdx != "":
        appdx = f"_{appdx}"

    arrayX_strides, yyyymmdd, _ = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                                dataToLoad = "limbX_strides", 
                                                appdx = appdx,
                                                limb = limbRef)
    
    arrayY_strides, _, _ = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                                dataToLoad = "limbY_strides", 
                                                yyyymmdd = yyyymmdd,
                                                appdx = appdx,
                                                limb = limbRef)
    
    df, _, _ = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                               dataToLoad = "strideParams", 
                                               yyyymmdd = yyyymmdd,
                                               appdx = appdx,
                                               limb = limbRef)   
            
    # PREPARE FOR TRAJECTORY PLOTTING
    # get param_split
    if param == 'headLVL':
        param_split = np.linspace(-40-0.00000001,40,group_num+1,endpoint = True)
        df[param] = [int(d[3:])*-1 for d in df[param]]
        
    elif param == 'snoutBodyAngle':
        if appdx == "":
            param_split = [141,147,154,161,167,174]
        else:
            param_split = [147,154,161,167,174]
            
    # find the median number of data points per stride 
    # the og arrays are padded with nans due to different numbers of per-stride data points
    ns = []
    for c in range(arrayX_strides.shape[1]):
        if not np.all(np.isnan(arrayX_strides[:,c])):
            ns.append(len(arrayX_strides[:,c][~np.isnan(arrayX_strides[:,c])]))
    med_length = np.median(ns).astype(int) 
    
    arrayX_resampled = np.empty((med_length, arrayX_strides.shape[1]))
    arrayY_resampled = np.empty((med_length, arrayY_strides.shape[1]))
    arrayX_resampled[:] = np.nan
    arrayY_resampled[:] = np.nan

    for og, new in zip([arrayX_strides, arrayY_strides],
                       [arrayX_resampled, arrayY_resampled]):
        for c in range(og.shape[1]):
            if np.all(np.isnan(og[:,c])):
                continue
            nonans = og[:,c][~np.isnan(og[:,c])]
            if len(nonans) > med_length:
                new[:,c] = utils_processing.downsample_data(nonans, med_length)
            elif len(nonans) < med_length:
                new[:,c] = np.interp(np.arange(med_length), np.linspace(0, med_length, len(nonans)).astype(int), nonans)
            else:
                new[:,c] = nonans
    
    grouped_dict = {'X':{}, 'Y':{}}
    for im, m in enumerate(Config.passiveOpto_config['mice']):
        df_sub = df[df['mouseID'] == m]
        df_grouped = df_sub.groupby(pd.cut(df_sub[param], param_split)) 
        group_row_ids = df_grouped.groups
        grouped_dict['X'][m] = {key: arrayX_resampled[:,val] for key,val in group_row_ids.items()} 
        grouped_dict['Y'][m] = {key: arrayY_resampled[:,val] for key,val in group_row_ids.items()}
       
    # invert the nested dictionary            
    grouped_dict_inv = utils_processing.invert_nested_dict(grouped_dict)
   
    import pickle
    pickle.dump(grouped_dict_inv, open(os.path.join(outputDir, f'{yyyymmdd}_limbKinematicsDict_{param}_{limbRef}{appdx}.pkl'), "wb" ))    

def get_egocentric_limb_positions(outputDir = Config.paths["mtTreadmill_output_folder"],
                                  yyyymmdd = '2022-05-06',
                                  param = None):
    df_limbs, yyyymmdd = data_loader.load_processed_data(outputDir = outputDir,
                                                   dataToLoad = 'mtLimbData',
                                                   yyyymmdd = yyyymmdd)
    df_others, _ = data_loader.load_processed_data(outputDir = outputDir,
                                                   dataToLoad = 'mtOtherData',
                                                   yyyymmdd = yyyymmdd)
    if param == 'snoutBodyAngle':
        bodyAngles, _ = data_loader.load_processed_data(outputDir=outputDir, 
                                                        dataToLoad = 'bodyAngles', 
                                                        yyyymmdd = yyyymmdd, 
                                                        appdx = '')
    
    lvl = 4 if '2021' in yyyymmdd else 5
    limbs = np.unique(df_limbs.columns.get_level_values(lvl))
    for i, limb in enumerate(limbs):
        for ic, coord in enumerate(['x','y']):
            if lvl == 5:
                df_limb_x = df_limbs.loc[:,(slice(None),slice(None),slice(None),slice(None),slice(None),limb,coord)]
            else:
                df_limb_x = df_limbs.loc[:,(slice(None),slice(None),slice(None),slice(None),limb,coord)]
            
            limb_x_means = np.mean(df_limb_x, axis = 0)
            
            if i == 0 and ic == 0:
                if lvl == 5:
                    df_body_x = df_others.loc[:,(slice(None),slice(None),slice(None),slice(None),slice(None),'body',coord)]
                else:
                    df_body_x = df_others.loc[:,(slice(None),slice(None),slice(None),slice(None),'body',coord)]
               
                body_x_means = np.mean(df_body_x, axis = 0)
                
                mouseIDs = df_limb_x.columns.get_level_values(0)
                expDates = df_limb_x.columns.get_level_values(1)
                trialNums = df_limb_x.columns.get_level_values(2)
                
                if param == 'snoutBodyAngle':
                    bodyAngles_deg = bodyAngles.loc[:, (slice(None),slice(None),slice(None),slice(None),slice(None),'snoutBody')]
                    trialTypes = np.mean(bodyAngles_deg, axis=0)
                else:
                    trialTypes = df_limb_x.columns.get_level_values(3)
                
                egocentricDF = pd.DataFrame(np.vstack((mouseIDs, expDates, trialNums,trialTypes)).T, 
                                            columns = ['mouseID', 'expDate', 'trialNum', 'trialType'])
            egocentricDF[f'{limb}{coord}'] = np.asarray(limb_x_means) - np.asarray(body_x_means)

    egocentricDF.to_csv(os.path.join(outputDir, f'{yyyymmdd}_limbPositionRegressionArray_egocentric.csv'))
    
def add_CoMy_to_strideParams(yyyymmdd,
                             appdx = '',
                             refLimb = 'lH1',
                             merged = False,
                             outputDir = Config.paths["passiveOpto_output_folder"]):
    """
    Takes a passive opto "strideParams" table and uses forceplate data to convert
    the "snoutBodyAngle" column into CoMy
    
    appdx should be supplied with a leading underscore (unless appdx = '')
    
    !!! overwrites the input file
    """
    
   # load passive opto data
    m = 'Merged' if merged else ''
    filepath = Path(outputDir)/f"{yyyymmdd}_strideParams{m}{appdx}_{refLimb}.csv"
    df = pd.read_csv(filepath)
    
    # load forceplate data
    tx = 'incline' if 'incline' in appdx else 'snoutBodyAngle'
    predictor = 'levels' if 'incline' in appdx else 'snoutBodyAngle'
    yyyymmdd_fp = Config.forceplate_config["passiveOpto_relations"][yyyymmdd][tx]
    filepath_fp = Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd_fp}_mixedEffectsModel_linear_COMy_{predictor}.csv"
    CoMy_pred = pd.read_csv(filepath_fp)
    CoMy_pred_intercept = CoMy_pred.iloc[0,1]
    CoMy_pred_slope = CoMy_pred.iloc[1,1]
    # load data that the model is based on for centering of sBA data
    df_fp = pd.read_csv(os.path.join(
        Config.paths["forceplate_output_folder"], f"{yyyymmdd_fp}_meanParamDF_{predictor}.csv")
        )
    
    comys = ((((df[tx]-np.nanmean(df_fp['param'])) * CoMy_pred_slope)
                + CoMy_pred_intercept)
                + np.nanmean(df_fp['CoMy_mean']))
    df["CoMy"] = comys
    df.to_csv(filepath)

def flip_homologous_phase(arr, col):
    # get mice based on their injection side
    mice_Linj = Config.injection_config['left_inj_imp']
    
    # flip phase of left-injected mice (assume injection on right side)
    arr.loc[
        arr['mouseID'].isin(mice_Linj),
        col,
        ] = arr.loc[
            arr['mouseID'].isin(mice_Linj),
            col
            ].apply(lambda s: -s)
    
    return arr
    
def compute_circ_corr(yyyymmdd, col1, col2,
                      outputDir=Config.paths["passiveOpto_output_folder"],
                      appdx='',
                      ref='lH1'):
    """
    This function implements Jammalamadaka-Sarma circ-circ correlation coefficient
        rho=1 : perfect positive circ corr, phases vary in sync
        rho=0 : no correlation, phases are independent
        rho=-1 : perfect negative circ corr, phases vary in opposite directions

    """
    
    import pycircstat as circ
    
    df, _, _ = data_loader.load_processed_data(outputDir = outputDir, 
                                                      yyyymmdd=yyyymmdd,
                                                      limb=ref,
                                                            dataToLoad = "strideParams", 
                                                            appdx = appdx)
    
    if 'COMBINED' in ref:
        raise ValueError('Function not implemented for combined ref limb datasets')
    
    df_sub = df[['mouseID',col1,col2]].copy()
    df_sub.dropna(inplace=True)
    
    # deal with unilateral inj
    if any('rH' in s for s in [col1, col2]) or any('rF' in s for s in [col1, col2]):
        # remove bilaterally injected mice
        df_sub = df_sub[~df_sub['mouseID'].isin(Config.injection_config['both_inj_left_imp'])]
        
        if 'rH' in col1 or 'rF' in col1:
            df_sub = flip_homologous_phase(df_sub, col1)
        if 'rH' in col1 or 'rF' in col2:
            df_sub = flip_homologous_phase(df_sub, col2)
    
    df_sub.drop(columns=['mouseID'], inplace=True)
    df_sub = df_sub*2*np.pi
    
    rho = circ.corrcc(df_sub[col1].to_numpy(), df_sub[col2].to_numpy()) 
    print(f"Circular correlation coefficient: {rho}")
    
    # compute_circ_corr(yyyymmdd='2022-08-18', col1='lF0', col2='rH0',ref='lH1',
    #                       outputDir=Config.paths["passiveOpto_output_folder"],
    #                       appdx='')
    # Circular correlation coefficient: -0.17197571324764552
    
    # compute_circ_corr(yyyymmdd='2022-08-18', col1='lF0', col2='rH0',ref='lH1',
    #                       outputDir=Config.paths["passiveOpto_output_folder"],
    #                       appdx='_incline')
    # Circular correlation coefficient: -0.16342120171320532
    
    # compute_circ_corr(yyyymmdd='2021-10-23', col1='lF0', col2='rH0',ref='lH1',
    #                       outputDir=Config.paths["mtTreadmill_output_folder"],
    #                       appdx='')
    # Circular correlation coefficient: 0.3873386913200928 
    
    # compute_circ_corr(yyyymmdd='2023-09-25', col1='lF0', col2='rH0',ref='lH1',
    #                       outputDir=Config.paths["mtTreadmill_output_folder"],
    #                       appdx='')
    # Circular correlation coefficient: 0.3975096246140336
    
    # compute_circ_corr(yyyymmdd='2022-05-06', col1='lF0', col2='rH0',ref='lH1',
    #                       outputDir=Config.paths["mtTreadmill_output_folder"],
    #                       appdx='')
    # Circular correlation coefficient: 0.38177089402548675
       
    
