"""
Functions specific to the motorised treamdill
"""
import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import copy
import itertools
import math
import pickle
import scipy.signal

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_processing, utils_math, mt_treadmill_data_manager
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def compute_mouse_speed(outputDir= Config.paths["mtTreadmill_output_folder"]):
    trdm_speed, yyyymmdd = data_loader.load_processed_data(outputDir, dataToLoad = 'treadmillSpeed')
    data, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'mtOtherData', yyyymmdd = yyyymmdd)    
    data_body = data.loc[:, (slice(None), slice(None), slice(None), slice(None), slice(None), 'body', 'x')]
    
    trdm_speed_filt = np.empty(trdm_speed.shape)
    data_body_filt = np.empty(data_body.shape)
    for col in range(data_body.shape[1]):
        # butter filter function takes care of nans
        trdm_speed_filt[:, col] = utils_processing.butter_filter(trdm_speed.iloc[:, col], filter_freq = 2, sampling_freq = Config.analysis_config["fps"])
        data_body_filt[:,col] = utils_processing.butter_filter(data_body.iloc[:, col], filter_freq = 2, sampling_freq = Config.analysis_config["fps"])
    data_body_filt_diff = np.diff(data_body_filt, axis = 0)*Config.analysis_config["fps"]/Config.analysis_config["px_per_cm"]["body"]
    mouse_speed = trdm_speed_filt[:-1, :] + data_body_filt_diff
    
    df = pd.DataFrame(mouse_speed, columns = trdm_speed.columns)
    
    df.to_csv(os.path.join(outputDir, yyyymmdd + '_mouseSpeed.csv'))
    
def get_relative_body_height(outputDir= Config.paths["mtTreadmill_output_folder"]):
    import statistics
        
    data_other, yyyymmdd = data_loader.load_processed_data(outputDir,
                                                           dataToLoad = 'mtOtherData')
    data_limbs, _ = data_loader.load_processed_data(outputDir, 
                                                    dataToLoad = 'mtLimbData',
                                                    yyyymmdd = yyyymmdd)
    mouse_speed, _ = data_loader.load_processed_data(outputDir, 
                                                     dataToLoad = 'mouseSpeed', 
                                                     yyyymmdd = yyyymmdd)
    
    if '2021' in yyyymmdd:
        d = 4
        tup_names = ['mouseID', 'expDate', 'trial', 'stimType']
    else:
        d = 5
        tup_names = ['mouseID', 'expDate', 'trial', 'trialType', 'stimType']
        
    data_tuples = mouse_speed.columns.values
    data_tuples = np.asarray([tup[:d] for tup in data_tuples]) # was :5 before incorporation of 'deg' level (5s for old data?)
    data_tuples = np.unique(data_tuples, axis = 0)
    
    if d == 4:
        data_body_y = data_other.loc[:, (slice(None), slice(None), slice(None), slice(None), 'body', 'y')]
        data_rH1_y = data_limbs.loc[:, (slice(None), slice(None), slice(None), slice(None), 'rH1', 'y')]
    else:
        data_body_y = data_other.loc[:, (slice(None), slice(None), slice(None), slice(None), slice(None), 'body', 'y')]
        data_rH1_y = data_limbs.loc[:, (slice(None), slice(None), slice(None), slice(None), slice(None), 'rH1', 'y')]
    
    body_relative = np.zeros_like(data_body_y) * np.nan
    new_tuples = []
    for c in range(mouse_speed.shape[1]):
        tup = mouse_speed.columns[c][:d]
        new_tuples.append(tup)
        body_y = np.asarray(data_body_y.loc[:,tup]).flatten()
        rH1_y = np.asarray(data_rH1_y.loc[:,tup]).flatten()
        is_locomoting = np.asarray(mouse_speed.loc[:,tup]).flatten()>5
        rH1_y_locom = utils_processing.remove_outliers(rH1_y[1:][is_locomoting])
        rH1_y_locom_mode = statistics.mode(rH1_y_locom)
        body_relative[:,c] = body_y-rH1_y_locom_mode  
    
    index = pd.MultiIndex.from_tuples(new_tuples, names = tup_names)
    body_relative_DF = pd.DataFrame(body_relative*-1, columns = index)
    body_relative_DF.to_csv(os.path.join(outputDir, yyyymmdd + f'_bodyHeight.csv')) 
    
def get_trdm_mouse_movement_boundaries(outputDir= Config.paths["mtTreadmill_output_folder"], yyyymmdd = None):
    """
    stores treadmill onset/offset and mouse locomotion onsets/offsets per trial in a nested dict
    """
    trdm_speed, yyyymmdd = data_loader.load_processed_data(outputDir, 
                                                           dataToLoad = 'treadmillSpeed',
                                                           yyyymmdd = yyyymmdd)
    mouse_speed, _ = data_loader.load_processed_data(outputDir, 
                                                     dataToLoad = 'mouseSpeed', 
                                                     yyyymmdd = yyyymmdd)
    other_data, _ = data_loader.load_processed_data(outputDir, 
                                                    dataToLoad = 'mtOtherData', 
                                                    yyyymmdd = yyyymmdd)
    
    trdm_threshold = 1
    locom_threshold = 5
    body_threshold = 450
    
    mvmt_dict = {}
    for col in range(mouse_speed.shape[1]):
        mouseID, expDate, trialNum, trialType, stimType = mt_treadmill_data_manager.get_metadata_from_df(mouse_speed, col)
        print(f"Processing column {col+1}/{mouse_speed.shape[1]}")
        bodyX = other_data.loc[:, (mouseID, expDate, trialNum, trialType, stimType, 'body', 'x')]
        trdm_speed_filt = utils_processing.butter_filter(trdm_speed.iloc[:, col], filter_freq = 2, sampling_freq = Config.analysis_config["fps"])
        
        # trdm movment ONSET when filtered speed is above thresh AND is increasing 
        # (to prevent inclusion of the decay of the trigger signal)
        trdmON = np.where((trdm_speed_filt >= trdm_threshold) & (np.concatenate(([0],np.diff(trdm_speed_filt))) > 0))[0][0]
        trdmOFF = np.where(trdm_speed_filt >= trdm_threshold)[0][-1]+1
        #trdmON and trdmOFF are single numbers because trdm movement is stable
        
        mouse_mvmt = np.where((mouse_speed.iloc[trdmON:trdmOFF, col] > locom_threshold) & (bodyX[trdmON:trdmOFF] > body_threshold))[0]+trdmON
        mvmt_switch_ids = np.where(np.diff(mouse_mvmt) > 1)[0]
        locomON = mouse_mvmt[np.where(np.diff(mouse_mvmt) > 1)[0]+1]
        locomOFF = mouse_mvmt[np.where(np.diff(mouse_mvmt) > 1)[0]]
        #add the first threshold crossing (ON) during treadmill movement
        if mouse_speed.iloc[trdmON,col] >locom_threshold:
            locomON = np.concatenate(([trdmON], locomON))
        else:
            locomON = np.concatenate(([mouse_mvmt[0]], locomON))
        # add the last threshold crossing (OFF) during treadmill movement
        if mouse_speed.iloc[trdmOFF, col] > locom_threshold:
            locomOFF = np.concatenate((locomOFF, [trdmOFF]))
        else:
            locomOFF = np.concatenate((locomOFF, [mouse_mvmt[-1]]))
        
        mvmt_dict = utils_processing.populate_nested_dict(targetDict = mvmt_dict, 
                             dataToAdd = (trdmON, trdmOFF, locomON, locomOFF), 
                             metadataLVL = (mouseID, expDate, trialNum, trialType, stimType))
    
    pickle.dump(mvmt_dict, open(os.path.join(outputDir, yyyymmdd+ '_movementDict.pkl'), "wb" ))      
    
def compute_locomotor_params(outputDir= Config.paths["mtTreadmill_output_folder"], 
                             ref_limb = 'lH1',
                             yyyymmdd = None):
    from scipy.stats import pearsonr
    limb_data, yyyymmdd = data_loader.load_processed_data(outputDir, 
                                                          dataToLoad = 'mtLimbData',
                                                          yyyymmdd = yyyymmdd) 
    other_data, yyyymmdd = data_loader.load_processed_data(outputDir, 
                                                          dataToLoad = 'mtOtherData',
                                                          yyyymmdd = yyyymmdd) 
    mouse_speed, _ = data_loader.load_processed_data(outputDir, 
                                                     dataToLoad = 'mouseSpeed', 
                                                     yyyymmdd = yyyymmdd)
    body_angles, _ = data_loader.load_processed_data(outputDir, 
                                                     dataToLoad = 'bodyAngles', 
                                                     yyyymmdd = yyyymmdd)
    body_heights, _ = data_loader.load_processed_data(outputDir, 
                                                    dataToLoad = 'bodyHeight', 
                                                    yyyymmdd = yyyymmdd)
    movement_dict, _ = data_loader.load_processed_data(outputDir, 
                                                       dataToLoad = 'movementDict', 
                                                       yyyymmdd = yyyymmdd)
    optotrig_dict, _ = data_loader.load_processed_data(outputDir, 
                                                       dataToLoad = 'optoTrigDict', 
                                                       yyyymmdd = yyyymmdd)
    
    lH1 = []; lH2 = []; lH0 = []
    rH1 = []; rH2 = []; rH0 = []
    lF1 = []; lF2 = []; lF0 = []
    rF1 = []; rF2 = []; rF0 = []
    
    if ref_limb == 'lH1':
        limb_list = [lH2, rH1, rH2, lF1, lF2, rF1, rF2]
        limb_list_str = ['lH2', 'rH1', 'rH2', 'lF1', 'lF2', 'rF1', 'rF2']
        mean_limb_list = [rH0, lF0, rF0] # for storage of mean of (rH1,rH2) etc
        mean_limb_str = ['rH0', 'lF0', 'rF0']
    elif ref_limb == 'rH1': 
        limb_list = [lH1, lH2, rH2, lF1, lF2, rF1, rF2]
        limb_list_str = ['lH1', 'lH2', 'rH2', 'lF1', 'lF2', 'rF1', 'rF2']
        mean_limb_list = [lH0, lF0, rF0] # for storage of mean of (lH1,lH2) etc
        mean_limb_str = ['lH0', 'lF0', 'rF0']
    else:
        raise ValueError("Invalid reference limb supplied!")
    
    xthr = 6
    mice = np.empty((0))
    expdates = np.empty((0))
    trialnums = np.empty((0))
    trialtypes = np.empty((0))
    stimtypes = np.empty((0))
    mice_long = np.empty((0))
    expdates_long = np.empty((0))
    trialnums_long = np.empty((0))
    trialtypes_long = np.empty((0))
    stimtypes_long = np.empty((0))
    stridenums = np.empty((0))    # count PER FILE -- over all bouts within a file
    stridelengths = np.empty((0))
    stridefreqs = np.empty((0))
    meanspeeds = np.empty((0))
    meansnoutbody = np.empty((0))
    meansnoutbodytail = np.empty((0))
    meanbodyheight = np.empty((0))
    bodyheight_osc = np.empty((0))
    
    # save npy arrays containing x and y coords during locomotion
    labels = ['lH1', 'lH2', 'rH1', 'rH2', 'lF1', 'lF2', 'rF1', 'rF2']#['lH1', 'rH1', 'lF1', 'rF1']
    array_x = np.empty((0, len(labels)))
    array_y = np.empty((0, len(labels)))
    array_speed = np.empty(0) # column = speed
    array_bodyAngles = np.empty((0)) # columns = bodySnout, snoutBodyTail
    array_bodyHeights_rel = np.empty((0))
    
    for col in range(mouse_speed.shape[1]):
        mouseID, expDate, trialNum, trialType, stimType = mt_treadmill_data_manager.get_metadata_from_df(mouse_speed, col)
        if '2021' in yyyymmdd:
            limb_data_sub = limb_data.loc[:, (mouseID, expDate, trialNum, trialType)]
            mouse_speed_sub = mouse_speed.loc[:, (mouseID, expDate, trialNum, trialType)]
            snoutbody_angles_sub = body_angles.loc[:, (mouseID, expDate, trialNum, trialType, 'snoutBody')]
            snoutbodytail_angles_sub = body_angles.loc[:, (mouseID, expDate, trialNum, trialType, 'snoutBodyTail')]
            bodyheight_rel_sub = body_heights.loc[:, (mouseID, expDate, trialNum, trialType)]
            bodyHeight_sub = other_data.loc[:, (mouseID, expDate, trialNum, trialType, 'body', 'y')]
            trdmON, trdmOFF, locomONs, locomOFFs = movement_dict[mouseID][expDate][trialNum][trialType]
        else:
            limb_data_sub = limb_data.loc[:, (mouseID, expDate, trialNum, trialType, stimType)]
            mouse_speed_sub = mouse_speed.loc[:, (mouseID, expDate, trialNum, trialType, stimType)]
            snoutbody_angles_sub = body_angles.loc[:, (mouseID, expDate, trialNum, trialType, stimType, 'snoutBody')]
            snoutbodytail_angles_sub = body_angles.loc[:, (mouseID, expDate, trialNum, trialType, stimType, 'snoutBodyTail')]
            bodyheight_rel_sub = body_heights.loc[:, (mouseID, expDate, trialNum, trialType, stimType)] 
            bodyHeight_sub = other_data.loc[:, (mouseID, expDate, trialNum, trialType, stimType, 'body', 'y')]
            trdmON, trdmOFF, locomONs, locomOFFs = movement_dict[mouseID][expDate][trialNum][trialType][stimType]
        print(f"Processing mouse {mouseID} from {expDate} trial {trialNum}...")
        
        refX = np.asarray(limb_data_sub.loc[:, (ref_limb, 'x')]).flatten()
        refX_filt = utils_processing.butter_filter(refX, filter_freq = 50, sampling_freq = Config.mtTreadmill_config['fps'])
        refY = np.asarray(limb_data_sub.loc[:, (ref_limb, 'y')]).flatten()
        refY_filt = utils_processing.butter_filter(refY, filter_freq = 50, sampling_freq = Config.mtTreadmill_config['fps'])
        
        for num, (on, off) in enumerate(zip(locomONs, locomOFFs)):
            print(f"Locomotor bout {num}")
            ypeaks = scipy.signal.find_peaks(refY_filt[on:off]*-1, prominence = 13)[0] # mid-swings
 #           print(f"ypeaks: {ypeaks}")
            if len(ypeaks) > 1:
                xpeaksfilt = np.empty(0) # stance onsets for all limbs
                xtroughsfilt = np.empty(0) # swing onsets for all limbs
                for i in range(ypeaks.shape[0]-1): # look for max and min x peak between every two y peaks, exactly one value by definition
  #                  print(f"ypeak #{i}")
                    xpeaksfilt_all = scipy.signal.find_peaks(refX_filt[on:off][ypeaks[i]:ypeaks[i+1]])[0]+ypeaks[i]
                    xtroughsfilt_all = scipy.signal.find_peaks(refX_filt[on:off][ypeaks[i]:ypeaks[i+1]]*-1)[0]+ypeaks[i]
                    if xpeaksfilt_all.shape[0] > 0 and xtroughsfilt_all.shape[0] > 0:
                        # print(xpeaksfilt)
                        xpeaksfilt = np.concatenate((xpeaksfilt, [xpeaksfilt_all[np.argmax(refX_filt[on:off][xpeaksfilt_all])]])).astype(int)
                        xtroughsfilt = np.concatenate((xtroughsfilt, [xtroughsfilt_all[np.argmin(refX_filt[on:off][xtroughsfilt_all])]])).astype(int)
                        # print(xpeaksfilt)
                    else:
                        print("Could not get x peaks!")
                        continue
                #arrays should be alternating by definition as long as ypeak detection works well
                if xpeaksfilt.shape[0] > 0 and xtroughsfilt.shape[0] > 0:
                    xpeaksfilt, xtroughsfilt = utils_processing.are_arrays_alternating(xpeaksfilt, xtroughsfilt) #returns alernating list of tr-pk---tr OR bools if things go wrong
                else:
                    print(f"No x peaks for any of {ypeaks.shape[0]-1} ypeak intervals!")
                    continue
               
                if type(xpeaksfilt) != bool :
                    xpeaks = []
                    xtroughs = []
                    [xpeaks.append(np.argmax(refX[on:off][(xpf-xthr):(xpf+xthr)])+xpf-xthr) for xpf in xpeaksfilt if (xpf >= xthr and xpf+xthr < len(refX[on:off]))]
                    for xpt in xtroughsfilt:
                        if (xpt >= xthr and xpt+xthr < len(refX[on:off])):
                            xtroughs.append(np.argmin(refX[on:off][(xpt-xthr):(xpt+xthr)])+xpt-xthr)
                        elif xpt >= xthr:
                            xtroughs.append(np.argmin(refX[(xpt-xthr):(len(refX[on:off])-1)])+xpt-xthr)
                        else:
                            xtroughs.append(np.argmin(refX[on:off][:(xpt+xthr)])) 
                else :
                    print("Could not get peaks!")
                    continue
                
                xpeaks = np.asarray(xpeaks)
                xtroughs = np.asarray(xtroughs)
                
                if xtroughs.shape[0] > 1 and xpeaks.shape[0] > 1:
                    for k in range(xtroughs.shape[0]-1):
                        stepStart = xtroughs[k]
                        stepEnd = xtroughs[k+1]
                        # print(stepStart, stepEnd)
                        dlc_limb = refX[on:off][stepStart:stepEnd] 
                        n = dlc_limb.shape[0]
                        delays = np.linspace(-0.5*n, 0.5*n, n).astype(int)
                        for bp, corr_arr in zip(limb_list_str, limb_list):
                            corrs = []
                            for t_delay in delays: # iterate over time delays
                                try:
                                    dlc = np.asarray(limb_data_sub.loc[on:off, (bp, 'x')][(stepStart+t_delay):(stepEnd+t_delay)]).flatten()
                                    corr, _ = pearsonr(dlc, dlc_limb)
                                    corrs.append(corr)
                                except:
  #                                  print('Out of range!')
                                    corrs.append(np.nan)
                            if not np.all(np.isnan(corrs)):
                                corr_arr.append(delays[np.nanargmax(corrs)] / delays.shape[0]) #appends a single value!
                            else:
                                corr_arr.append(np.nan)
                        
                        # matched speed and snout-body angles                   
                        meanspeeds = np.append(meanspeeds, np.mean(mouse_speed_sub[on:off][stepStart:stepEnd]))
                        meansnoutbody = np.append(meansnoutbody, np.mean(snoutbody_angles_sub[on:off][stepStart:stepEnd]))
                        meansnoutbodytail = np.append(meansnoutbodytail, np.mean(snoutbodytail_angles_sub[on:off][stepStart:stepEnd]))
                        meanbodyheight = np.append(meanbodyheight, np.mean(bodyheight_rel_sub[on:off][stepStart:stepEnd]))
                        bodyheight_osc = np.append(bodyheight_osc, np.max(bodyHeight_sub[on:off][stepStart:stepEnd]) - np.min(bodyHeight_sub[on:off][stepStart:stepEnd]))
                    
                    stride_freqs = 1/(np.diff(xtroughs)/Config.mtTreadmill_config['fps']) #the number of data points from swing-on to swing-on
                    if xtroughs.shape[0] > xpeaks.shape[0]: #stride_freqs as long as stride_lengths
                        stride_lengths = (np.asarray(limb_data_sub.loc[on:off, (ref_limb, 'x')])[xpeaks]-np.asarray(limb_data_sub.loc[on:off, (ref_limb, 'x')])[xtroughs[:-1]])/Config.mtTreadmill_config["px_per_cm"][ref_limb]
                    elif xtroughs.shape[0] == xpeaks.shape[0]: #stride_freqs as long as stride_lengths
                        stride_freqs = np.append(stride_freqs, np.nan)
                        meanspeeds = np.append(meanspeeds, np.nan)
                        meansnoutbody = np.append(meansnoutbody, np.nan)
                        meansnoutbodytail = np.append(meansnoutbodytail, np.nan)
                        meanbodyheight = np.append(meanbodyheight, np.nan)
                        bodyheight_osc = np.append(bodyheight_osc, np.nan)
                        for corr_arr in limb_list:
                            corr_arr.append(np.nan)
                        stride_lengths = (np.asarray(limb_data_sub.loc[on:off, (ref_limb, 'x')])[xpeaks]-np.asarray(limb_data_sub.loc[on:off, (ref_limb,'x')])[xtroughs])/Config.mtTreadmill_config["px_per_cm"][ref_limb]
                    else:
                        print(f"More peaks than troughs! Peaks: {len(xpeaks)} Troughs: {len(xtroughs)}")

                else:
                    print('Peaks and troughs do not have a length >1 ! Discarding...')
                    continue
                
                # populate the x and y arrays
                array_x = np.append(array_x, np.asarray(limb_data_sub.loc[on:off, (labels, 'x')]), axis = 0)
                array_y = np.append(array_y, np.asarray(limb_data_sub.loc[on:off, (labels, 'y')]), axis = 0)
                array_speed = np.append(array_speed, np.asarray(mouse_speed_sub.loc[on:off]))
                array_bodyAngles = np.append(array_bodyAngles, np.asarray(snoutbody_angles_sub.loc[on:off]))
                array_bodyHeights_rel = np.append(array_bodyHeights_rel, np.asarray(bodyheight_rel_sub.loc[on:off]))
                mice_long = np.append(mice_long, np.repeat(mouseID, len(limb_data_sub.loc[on:off, (labels, 'x')])))
                expdates_long =  np.append(expdates_long, np.repeat(expDate, len(limb_data_sub.loc[on:off, (labels, 'x')])))
                trialnums_long =  np.append(trialnums_long, np.repeat(trialNum, len(limb_data_sub.loc[on:off, (labels, 'x')])))
                trialtypes_long =  np.append(trialtypes_long, np.repeat(trialType, len(limb_data_sub.loc[on:off, (labels, 'x')])))
                stimtypes_long =  np.append(stimtypes_long, np.repeat(stimType, len(limb_data_sub.loc[on:off, (labels, 'x')])))
                
            else:
                print('Zero y peaks! Discarding...')
                continue
            
            stridelengths = np.append(stridelengths, stride_lengths)
            stridefreqs = np.append(stridefreqs, stride_freqs)
            mice = np.append(mice, np.repeat(mouseID, stride_lengths.shape[0]))
            expdates = np.append(expdates, np.repeat(expDate, stride_lengths.shape[0]))
            trialnums = np.append(trialnums, np.repeat(trialNum, stride_lengths.shape[0]))
            trialtypes = np.append(trialtypes, np.repeat(trialType, stride_lengths.shape[0]))
            stimtypes = np.append(stimtypes, np.repeat(stimType, stride_lengths.shape[0]))
            stridenums = np.append(stridenums, np.arange(stride_lengths.shape[0])+1) 
                
    # COMPUTE MEAN LIMB PHASES + CORRECT FOR INCONSISTENT TRACKING & WALL-HOLDING WITH NANs
    for mean_limb_arr, corr_limb_arr in zip(mean_limb_list,[limb_list[1:3], limb_list[3:5], limb_list[5:7]]) :
        diff = abs(np.asarray(corr_limb_arr[0])-np.asarray(corr_limb_arr[1])) # difference between the two markers
        mean_corr = np.mean(corr_limb_arr, axis = 0) # NOT nanmean: if one is nan, the mean should be nan (discard that stride!)
        mean_corr[diff>0.05] = np.nan # if difference between the two markers big, at least one is poorly tracked
        [mean_limb_arr.append(m) for m in mean_corr]
    
    # mice = utils_processing.flatten_list(mice)
    # expdates = utils_processing.flatten_list(expdates)
    # trialnums = utils_processing.flatten_list(trialnums)
    # trialtypes = utils_processing.flatten_list(trialtypes)
    # stimtypes = utils_processing.flatten_list(stimtypes)
        
    array_identifier = np.vstack((mice_long, expdates_long, trialnums_long, trialtypes_long, stimtypes_long)).T      
    np.save(Path(outputDir)/ (yyyymmdd + "_arrayIdentifier.npy"), array_identifier) 
            
    df = pd.DataFrame(np.vstack((mice, expdates, trialnums, trialtypes, stimtypes, stridenums, 
                                 stridelengths, stridefreqs, limb_list[0], limb_list[1], 
                                 limb_list[2], limb_list[3], limb_list[4], limb_list[5], 
                                 limb_list[6], mean_limb_list[0], mean_limb_list[1],
                                 mean_limb_list[2], meanspeeds, meansnoutbody, meansnoutbodytail,
                                 meanbodyheight, bodyheight_osc)).T,
                      columns = ['mouseID', 'expDate', 'trialNum', 'trialType','stimType', 'strideNum', 
                                 'strideLength', 'strideFreq',  limb_list_str[0], 
                                 limb_list_str[1], limb_list_str[2], limb_list_str[3], 
                                 limb_list_str[4], limb_list_str[5], limb_list_str[6],
                                 mean_limb_str[0], mean_limb_str[1], mean_limb_str[2],
                                 'speed', 'snoutBodyAngle', 'snoutBodyTailAngle', 'bodyHeight_rel','bodyHeightOscillation'])
    df.to_csv(os.path.join(outputDir, yyyymmdd + f'_strideParams_{ref_limb}.csv'))
    
    np.save(Path(outputDir)/ (yyyymmdd + "_limbX.npy"), array_x)  #   'lH1', 'rH1', 'lF1', 'rF1'
    np.save(Path(outputDir)/ (yyyymmdd + "_limbY.npy"), array_y)  #   'lH1', 'rH1', 'lF1', 'rF1'  
    np.save(Path(outputDir)/ (yyyymmdd + "_limbX_speed.npy"), array_speed) 
    np.save(Path(outputDir)/ (yyyymmdd + "_limbX_bodyAngles.npy"), array_bodyAngles) 
    np.save(Path(outputDir)/ (yyyymmdd + "_limbX_bodyHeights_rel.npy"), array_bodyHeights_rel) 
    