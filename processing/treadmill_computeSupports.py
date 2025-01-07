import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.signal

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_processing, utils_math
from processing.data_config import Config

def compute_supports(arr):
    """
    computes fractions of 0,1,2,3,4-limb support per stride
    
    arr is an array of zeros (swings) and ones (stances)
    arr is limbs x time
    limbs are lH, rH, lF, rF
    """
    # assumes that reference limb data is in row 0
    ref_stride_onsets = np.concatenate(([0],
                                        np.where(np.diff(arr[0,:]) == -1)[0],
                                        [arr.shape[1]-1]))
    frac0 = []
    frac1 = []
    frac2diag = []
    frac2hmlt = []
    frac2hmlgFORE = []
    frac2hmlgHIND = []
    frac3 = []
    frac4 = []
    for i in range(len(ref_stride_onsets)-1):
        x = np.sum(arr[:, ref_stride_onsets[i]:ref_stride_onsets[i+1]], axis =0)
        frac0.append((x == 0).sum()/len(x))
        frac1.append((x == 1).sum()/len(x))
        frac3.append((x == 3).sum()/len(x))
        frac4.append((x == 4).sum()/len(x))
        x2 = np.where(x == 2)[0]
        arr_sub = arr[:, ref_stride_onsets[i]:ref_stride_onsets[i+1]][:, x2]
        diag = (((arr_sub[0,:] == 1) & (arr_sub[3,:] == 1)).sum() + 
                         ((arr_sub[1,:] == 1) & (arr_sub[2,:] == 1)).sum())
        hmlt = (((arr_sub[0,:] == 1) & (arr_sub[2,:] == 1)).sum() + 
                         ((arr_sub[1,:] == 1) & (arr_sub[3,:] == 1)).sum())
        hmlgHIND = ((arr_sub[0,:] == 1) & (arr_sub[1,:] == 1)).sum() 
        hmlgFORE = ((arr_sub[2,:] == 1) & (arr_sub[3,:] == 1)).sum()
        # other = arr_sub.shape[1] - diag
        frac2diag.append(diag / len(x))
        frac2hmlt.append(hmlt / len(x))
        frac2hmlgHIND.append(hmlgHIND / len(x))
        frac2hmlgFORE.append(hmlgFORE / len(x))
        # frac2other.append(other / len(x))
    
    return (frac0, frac1, frac2diag, frac2hmlt, frac2hmlgHIND, frac2hmlgFORE, frac3, frac4)

def compute_duty_cycles(arr, *args):
    """
    computes duty cycles per stride for each limb
    arr is limbs x time
    limbs are lH, rH, lF, rF
    
    args are other Tx1 arrays containing metadata (speed, snoutBodyAngle, mouseID, etc)
    
    returns a numpy array (stride_num x 2+len(args))
    first column is the duty cycle, second column is limb str, the rest are args in order
    """
    lH_duties = []
    rH_duties = []
    lF_duties = []
    rF_duties = []
    limbs = ['lH','rH','lF','rF']
    
    stride_onsets = np.concatenate(([0],
                                    np.where(np.diff(arr[0,:]) == -1)[0],
                                    [arr.shape[1]-1]))
    
    duty_grande = np.empty((0, len(args)+2))
    for ilimb, duty_list in enumerate([lH_duties, rH_duties, lF_duties, rF_duties]):
        # stride_onsets = np.concatenate(([0],
        #                                 np.where(np.diff(arr[ilimb,:]) == -1)[0],
        #                                 [arr.shape[1]-1]))
        
        for i in range(len(stride_onsets)-1):
               stride = arr[ilimb, stride_onsets[i]:stride_onsets[i+1]] 
               duty_list.append((stride == 1).sum()/len(stride))
        
        duty_arr = np.asarray(duty_list)
        duty_arr = np.vstack((duty_arr, np.repeat(limbs[ilimb], repeats = len(duty_list)))).T
        for j, arg in enumerate(args):
            arg_list = []
            for i in range(len(stride_onsets)-1):
                arg_list.append(np.nanmean(arg[stride_onsets[i]:stride_onsets[i+1]]))
            duty_arr = np.hstack((duty_arr, np.asarray(arg_list).reshape(-1,1)))
        duty_grande = np.concatenate((duty_grande, duty_arr), axis = 0)

    return duty_grande

def get_support_fractions_passiveOpto(limb = 'lH1', 
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
                                                    appdx = appdx) #'snoutBody' 'snoutBodyTail'
    if os.path.exists(Path(outputDir)/"metadataPyRAT_processed.csv"):
        metadata_df = data_loader.load_processed_data(outputDir, 
                                                      dataToLoad = 'metadataProcessed', 
                                                      yyyymmdd = yyyymmdd, 
                                                      appdx = appdx)
        if 'rl' in arrayIDer[0,3]:
            headHeight_scaled = [(-int(x[2:])+24.5)/Config.passiveOpto_config["mm_per_g"] for x in arrayIDer[:,3]]
    
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
    
    if outputDir == Config.paths["mtTreadmill_output_folder"]:
        limb_id_dict = {'lH1': 0, 'lH2': 1, 'rH1': 2, 'rH2': 3, 
                        'lF1': 4, 'lF2': 5, 'rF1': 6, 'rF2': 7}
        fps = Config.mtTreadmill_config["fps"]
    else:
        limb_id_dict = {'lH1': 0, 'lH2': 1, 'rH1': 2, 'rH2': 3, 
                        'lF1': 4, 'lF2': 5, 'rF1': 6, 'rF2': 7} #cannot be changed because this is the order in the limbX file
        bodyAngles = bodyAngles[:, 0]
        fps = Config.passiveOpto_config["fps"]
        
    problem = 0
    
    mouseID = np.empty((0))
    expDate = np.empty((0))
    stimFreq = np.empty((0))
    headLVL = np.empty((0))
    
    strideLength = np.empty((0))
    strideFreq = np.empty((0))
    strideNum = np.empty((0))
    sexes = np.empty((0))
    weights = np.empty((0))
    ages = np.empty((0))
    headHW = np.empty((0)) # head heights scaled by the average size of female/male mice
    speed_arr = []
    accel_arr = []
    snoutBody_arr = []
    bodyAngleRange_arr = []
    frac0arr = np.empty((0))
    frac1arr = np.empty((0))
    frac2diagarr = np.empty((0))
    frac2hmltarr = np.empty((0))
    frac2hmlgHINDarr = np.empty((0))
    frac2hmlgFOREarr = np.empty((0))
    frac3arr = np.empty((0))
    frac4arr = np.empty((0))
    duty_lH_arr = np.empty((0))
    duty_rH_arr = np.empty((0))
    duty_lF_arr = np.empty((0))
    duty_rF_arr = np.empty((0))
    # duty_arr_grande = np.empty((0, 9))
    lH1 = []; lH2 = []; lH0 = []
    rH1 = []; rH2 = []; rH0 = []
    lF1 = []; lF2 = []; lF0 = []
    rF1 = []; rF2 = []; rF0 = []
    
    if limb == 'lH1':
        limb_list_str_supp = ['rH1', 'lF1', 'rF1']
        limb_list = [lH2, rH1, rH2, lF1, lF2, rF1, rF2]
        limb_list_str = ['lH2', 'rH1', 'rH2', 'lF1', 'lF2', 'rF1', 'rF2']
        mean_limb_list = [rH0, lF0, rF0] # for storage of mean of (rH1,rH2) etc
        mean_limb_str = ['rH0', 'lF0', 'rF0']
    elif limb == 'rH1':
        limb_list_str_supp = ['lH1', 'lF1', 'rF1']
        limb_list = [rH2, lH1, lH2, lF1, lF2, rF1, rF2]
        limb_list_str = ['rH2', 'lH1', 'lH2', 'lF1', 'lF2', 'rF1', 'rF2']
        mean_limb_list = [lH0, lF0, rF0] # for storage of mean of (lH1,lH2) etc
        mean_limb_str = ['lH0', 'lF0', 'rF0']
    
    stride_pt_max = Config.passiveOpto_config['fps']
    arrayX_strides = np.empty((stride_pt_max,0)) # for coordinate data per stride
    arrayX_strides2 = np.empty((stride_pt_max,0))
    arrayY_strides = np.empty((stride_pt_max,0))
    
    # consider each constituent file separately - ONLY lH1 or rH1 for now
    # use the filtered data to identify the number of peaks needed and find raw peaks in vicinity
    for fon, foff in zip(file_borders[:-1], file_borders[1:]):
    # for fon, foff in zip(file_borders[772:-1], file_borders[773:]): #131-133 # 150-153
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
            peaks_true = []
            troughs_true = []
            [peaks_true.append(np.argmax(arr_seg[(pf-13):(pf+13)])+pf-13) for pf in peaks_filt if (pf >= 13 and pf+13 < len(arrayX[fon:foff, limb_id_dict[limb]]))]
            for pt in troughs_filt:
                if (pt >= 13 and pt+13 < len(arrayX[fon:foff, limb_id_dict[limb]])):
                    troughs_true.append(np.argmin(arr_seg[(pt-13):(pt+13)])+pt-13)
                elif pt >= 13:
                    troughs_true.append(np.argmin(arr_seg[(pt-13):(len(arrayX[fon:foff, 0])-1)])+pt-13)
                else:
                    troughs_true.append(np.argmin(arr_seg[:(pt+13)])) 
            
            peaks_true = np.asarray(peaks_true)
            troughs_true =np.asarray(troughs_true)
        else:
            print(f"Could not get peaks from {'_'.join(arrayIDer[fon,:])}!")
            continue
        
        # check that peaks/troughs are alternating
        arr = np.concatenate((np.vstack((troughs_true, np.zeros_like(troughs_true))), 
                              np.vstack((peaks_true, np.zeros_like(peaks_true)+1))), axis = 1).T
        condition = utils_processing.alternating(arr[np.argsort(arr[:,0])][:,1])
       
        # iterate over strides
        if troughs_true.shape[0] > 1 and peaks_true.shape[0] > 1 and condition:
            stride_storage = np.zeros((len(limb_list_str_supp)+1, troughs_true[-1]-troughs_true[0])) * np.nan 
            for i in range(troughs_true.shape[0]-1):
                stepStart = troughs_true[i]
                stepEnd = troughs_true[i+1]
                
                # append ref limb peaks(= stance onset = 1)/troughs(= swing onset = 0)
                stride_storage[0,(troughs_true[i]-troughs_true[0]):(troughs_true[i+1]-troughs_true[0])] = np.concatenate((np.zeros(peaks_true[i]-troughs_true[i]),
                                                                                                                          np.ones(troughs_true[i+1]-peaks_true[i])))
               
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
                        
                # matched speed and snout-body angles SUPPORTS
                # speed_arr.append(np.nanmean(speed[fon:foff][stepStart:stepEnd]))
                # accel_arr.append(np.nanmean(np.gradient(speed[fon:foff][stepStart:stepEnd])*fps))
                # snoutBody_arr.append(np.nanmean(bodyAngles[fon:foff][stepStart:stepEnd]))
                # bodyAngleRange_arr.append(np.nanmax(bodyAngles[fon:foff][stepStart:stepEnd]) - np.nanmin(bodyAngles[fon:foff][stepStart:stepEnd]))
                
                # matched speed and snout-body angles GAIT PARAMS
                d = delays[0] if abs(delays[0])<=stepStart else 0
                d2 = delays[-1] if (stepEnd+delays[-1])>=(foff-fon) else (foff-fon)
                speed_arr.append(np.nanmean(speed[fon:foff][(stepStart+d):(stepEnd+d2)]))
                accel_arr.append(np.nanmean(np.gradient(speed[fon:foff][(stepStart+d):(stepEnd+d2)])*fps))
                snoutBody_arr.append(np.nanmean(bodyAngles[fon:foff][(stepStart+d):(stepEnd+d2)]))
                if len(bodyAngles[fon:foff][(stepStart+d):(stepEnd+delays[-1])]) > 0:
                    bodyAngleRange_arr.append(np.nanmax(bodyAngles[fon:foff][(stepStart+d):(stepEnd+d2)]) - np.nanmin(bodyAngles[fon:foff][(stepStart+d):(stepEnd+d2)]))
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
                    

            for ibp, bp in enumerate(limb_list_str_supp): # iterate over non-reference bps for supports!!
                arrX_filt_seg_bp = arrayX_filtered[fon:foff, limb_id_dict[bp]]
                arr_seg_bp = arrayX[fon:foff, limb_id_dict[bp]]
                peaks_filt_bp = scipy.signal.find_peaks(arrX_filt_seg_bp, prominence = 50)[0]
                troughs_filt_bp = scipy.signal.find_peaks(arrX_filt_seg_bp*-1, prominence = 50)[0]
                if peaks_filt_bp.shape[0] == 0 or troughs_filt_bp.shape[0] == 0: # no steps detected (there should always be locomotion, but some fragments are just too short to have steps)
                    print("No peaks and troughs detected!")    
                    continue
                peaks_filt_bp, troughs_filt_bp = utils_processing.are_arrays_alternating(peaks_filt_bp, troughs_filt_bp)
                if type(peaks_filt_bp) != bool :
                    peaks_true_bp = []
                    troughs_true_bp = []
                    [peaks_true_bp.append(np.argmax(arr_seg_bp[(pf-13):(pf+13)])+pf-13) for pf in peaks_filt_bp if (pf >= 13 and pf+13 < len(arrayX[fon:foff, limb_id_dict[bp]]))]
                    for pt in troughs_filt_bp:
                        if (pt >= 13 and pt+13 < len(arrayX[fon:foff, limb_id_dict[bp]])):
                            troughs_true_bp.append(np.argmin(arr_seg_bp[(pt-13):(pt+13)])+pt-13)
                        elif pt >= 13:
                            troughs_true_bp.append(np.argmin(arr_seg_bp[(pt-13):(len(arrayX[fon:foff, 0])-1)])+pt-13)
                        else:
                            troughs_true_bp.append(np.argmin(arr_seg_bp[:(pt+13)])) 
                    
                    peaks_true_bp = np.asarray(peaks_true_bp)
                    troughs_true_bp =np.asarray(troughs_true_bp)
                else:
                    print(f"Could not get peaks from {'_'.join(arrayIDer[fon,:])}!")
                    continue
                
                # set concatenation order
                rev_concat_order= False            
                
                # iterate over strides
                try:
                # if troughs_true_bp.shape[0] > 1 and peaks_true_bp.shape[0] > 1 and condition:
                    # IF THE FIRST BP-LIMB TROUGH and or peak IS BEFORE THE FIRST 
                    # REF-LIMB TROUGH (i.e. outside the window of interest)
                    troughs_to_remove = np.where(troughs_true_bp < troughs_true[0])[0]
                    peaks_to_remove = np.where(peaks_true_bp < troughs_true[0])[0]
                    if len(troughs_to_remove) > 0:
                        print(f"First {bp} trough before the first {limb} trough... remove!")
                        troughs_true_bp = troughs_true_bp[(troughs_to_remove[-1]+1):]
                        if len(peaks_to_remove) > 0: # cannot have peaks to remove if no troughs to remove because peak[0]>trough[0]
                            print(f"First {bp} peak before the first {limb} trough... remove!")
                            peaks_true_bp = peaks_true_bp[(peaks_to_remove[-1]+1):]
                    if troughs_true_bp[0] < peaks_true_bp[0] :    
                        #start in swing
                        peaks_true_bp = np.insert(peaks_true_bp, 0, troughs_true[0])
                        rev_concat_order = True
                    else:
                        #start in stance
                        troughs_true_bp = np.insert(troughs_true_bp, 0, troughs_true[0])
                
                    # IF THE LAST BP-LIMB TROUGH and or peak IS AFTER THE LAST
                    # REF-LIMB TROUGH (i.e. outside the window of interest)
                    troughs_to_remove = np.where(troughs_true_bp>troughs_true[-1])[0]
                    peaks_to_remove = np.where(peaks_true_bp>troughs_true[-1])[0]
                    if len(troughs_to_remove) > 0:
                        print(f"Last {bp} trough after the last {limb} trough... remove!")
                        troughs_true_bp = troughs_true_bp[:troughs_to_remove[0]] 
                    if len(peaks_to_remove) > 0:
                        print(f"Last {bp} peak after the last {limb} trough... remove!")
                        peaks_true_bp = peaks_true_bp[:peaks_to_remove[0]] 
                    if troughs_true_bp[-1] > peaks_true_bp[-1] :    
                        #end in stance
                        peaks_true_bp = np.append(peaks_true_bp, troughs_true[-1])
                    else:
                        #end in swing
                        troughs_true_bp = np.append(troughs_true_bp, troughs_true[-1])
                
                    combined_points_bp = np.sort(np.concatenate((troughs_true_bp, peaks_true_bp)))
                    for i in range(len(combined_points_bp)-1):
                        start = combined_points_bp[i]
                        end = combined_points_bp[i+1]
                        a = start - combined_points_bp[0]
                        b = end - combined_points_bp[0]
                        
                        if rev_concat_order:
                            if i % 2 == 0:
                                stride_storage[ibp+1, a:b] = np.ones(end-start)
                            else:
                                stride_storage[ibp+1, a:b] = np.zeros(end-start)
                        else:
                            if i % 2 == 0:
                                stride_storage[ibp+1, a:b] = np.zeros(end-start)
                            else:
                                stride_storage[ibp+1, a:b] = np.ones(end-start)
                except:
                    arr = np.concatenate((np.vstack((troughs_true, np.zeros_like(troughs_true))), 
                                          np.vstack((peaks_true, np.zeros_like(peaks_true)+1))), axis = 1).T
                    condition = utils_processing.alternating(arr[np.argsort(arr[:,0])][:,1])
                    if len(troughs_true_bp) == 0 or len(peaks_true_bp) == 0:
                        print(f"Zero troughs | peaks remain after cleaning! Excluding {fon}---{foff}... ")
                    # check that peaks/troughs are alternating
                    elif not condition:
                        print(f"Troughs/peaks not alternating! Excluding {fon}---{foff}... ") 
                    else:
                        print(f"Losing too many peaks while cleaning! Excluding {fon}---{foff}...")
                    continue
            
            fx = 0
            fxlist = []
            # GAIT PARAMS
            stride_freqs = 1/(np.diff(troughs_true)/Config.passiveOpto_config['fps']) #the number of data points from swing-on to swing-on
            if troughs_true.shape[0] > peaks_true.shape[0]: #stride_freqs as long as stride_lengths
                stride_lengths = (arr_seg[peaks_true]-arr_seg[troughs_true[:-1]])/Config.passiveOpto_config["px_per_cm"][limb]
            
            elif troughs_true.shape[0] == peaks_true.shape[0]: #stride_freqs shorten than? stride_lengths
                stride_freqs = np.append(stride_freqs, np.nan)
                speed_arr.append(np.nan)
                accel_arr.append(np.nan)
                snoutBody_arr.append(np.nan)
                bodyAngleRange_arr.append(np.nan)
                arrayX_strides = np.append(arrayX_strides, (np.zeros(stride_pt_max) + np.nan).reshape(-1,1), axis = 1)
                arrayX_strides2 = np.append(arrayX_strides2, (np.zeros(stride_pt_max) + np.nan).reshape(-1,1), axis = 1)
                arrayY_strides = np.append(arrayY_strides, (np.zeros(stride_pt_max) + np.nan).reshape(-1,1), axis = 1)
                
                for corr_arr in limb_list:
                    corr_arr.append(np.nan)
                
                fx = 1
                fxlist = [np.nan]
            
                stride_lengths = (arr_seg[peaks_true]-arrayX[fon:foff, limb_id_dict[limb]][troughs_true])/Config.passiveOpto_config["px_per_cm"][limb]
            else:
                print(f"More peaks than troughs! Peaks: {len(peaks_true)} Troughs: {len(troughs_true)}")
                # continue

            
            # SUPPORT
            if np.isnan(stride_storage).sum() != 0:
                print(f"Nans found! Excluding {fon}---{foff}...")
                speed_arr = speed_arr[:-(troughs_true.shape[0]-int(not(fx)))]
                accel_arr = accel_arr[:-(troughs_true.shape[0]-int(not(fx)))]
                snoutBody_arr = snoutBody_arr[:-(troughs_true.shape[0]-int(not(fx)))]
                bodyAngleRange_arr = bodyAngleRange_arr[:-(troughs_true.shape[0]-int(not(fx)))]
                for icr in range(len(limb_list)):
                    limb_list[icr] = limb_list[icr][:-(troughs_true.shape[0]-int(not(fx)))]
                # frac0 = frac1 = frac2diag = frac2other = frac3 = frac4 = [np.nan]
                print(f"{fon}, {foff} LENGTHS ARE: {len(mouseID)}, {len(limb_list[0])}")
                continue

            frac0, frac1, frac2diag, frac2hmlt, frac2hmlgHIND, frac2hmlgFORE, frac3, frac4 = compute_supports(stride_storage)
            frac0arr = np.concatenate((frac0arr, frac0, fxlist))
            frac1arr = np.concatenate((frac1arr, frac1, fxlist))
            frac2diagarr = np.concatenate((frac2diagarr, frac2diag, fxlist))
            frac2hmltarr = np.concatenate((frac2hmltarr, frac2hmlt, fxlist))
            frac2hmlgHINDarr = np.concatenate((frac2hmlgHINDarr, frac2hmlgHIND, fxlist))
            frac2hmlgFOREarr = np.concatenate((frac2hmlgFOREarr, frac2hmlgFORE, fxlist))
            frac3arr = np.concatenate((frac3arr, frac3, fxlist))
            frac4arr = np.concatenate((frac4arr, frac4, fxlist))
                            
            mouseID = np.append(mouseID, np.repeat(arrayIDer[fon,0], len(frac0)+fx))
            expDate = np.append(expDate, np.repeat(arrayIDer[fon,1], len(frac0)+fx))
            stimFreq = np.append(stimFreq, np.repeat(arrayIDer[fon,2], len(frac0)+fx))
            headLVL = np.append(headLVL, np.repeat(arrayIDer[fon,3], len(frac0)+fx))
            strideNum = np.append(strideNum, np.arange(len(frac0)+fx)+1) 
            
            duty_arr = compute_duty_cycles(stride_storage, 
                                           speed[fon:foff],
                                           np.gradient(speed[fon:foff])*fps,
                                           bodyAngles[fon:foff])
            for g in range(4):
                duty_arr = np.hstack((duty_arr, np.repeat(arrayIDer[fon,g], duty_arr.shape[0]).reshape(-1,1)))
            duty_lH_arr = np.concatenate((duty_lH_arr, duty_arr[duty_arr[:,1]=='lH'][:,0], fxlist))
            duty_rH_arr = np.concatenate((duty_rH_arr, duty_arr[duty_arr[:,1]=='rH'][:,0], fxlist))
            duty_lF_arr = np.concatenate((duty_lF_arr, duty_arr[duty_arr[:,1]=='lF'][:,0], fxlist))
            duty_rF_arr = np.concatenate((duty_rF_arr, duty_arr[duty_arr[:,1]=='rF'][:,0], fxlist))
            # duty_arr_grande = np.concatenate((duty_arr_grande, duty_arr))
            # duty_arr_grande is initialised to have shape (nrows,8);
            # if more args are fed to compute_duty_cycles, this needs to be changed!
            
            strideLength = np.concatenate((strideLength, stride_lengths))
            strideFreq = np.concatenate((strideFreq, stride_freqs))
            
            if os.path.exists(Path(outputDir)/"metadataPyRAT_processed.csv"):
                weight, age = utils_processing.get_weight_age(metadata_df, arrayIDer[fon,0], arrayIDer[fon,1])
                weights = np.append(weights, np.repeat(weight, len(frac0)+fx))
                ages = np.append(ages, np.repeat(age, len(frac0)+fx))
                if np.asarray(metadata_df[metadata_df['mouseID']['mouseID']==arrayIDer[fon,0]]['Sex']['Sex'])[0] == 'f':
                    sexes = np.append(sexes, np.repeat(1, len(frac0)+fx)) # append 1 if FEMALE
                else:
                    sexes = np.append(sexes, np.repeat(0, len(frac0)+fx)) # append 0 if MALE
                
                if 'rl' in arrayIDer[0,3]:
                    headHW = np.append(headHW, np.repeat(headHeight_scaled[fon]/weight, len(frac0)+fx))   
            if len(mouseID) != len(limb_list[0]):
                print(f"{fon}, {foff} LENGTHS ARE: {len(mouseID)}, {len(limb_list[0])}")
                break
        else:
            (f'PROBLEM WITH PEAK ASSIGNMENT! EXCLUDING {fon}---{foff}...')
    
    
    # COMPUTE MEAN LIMB PHASES + CORRECT FOR INCONSISTENT TRACKING & WALL-HOLDING WITH NANs
    for mean_limb_arr, corr_limb_arr in zip(mean_limb_list,[limb_list[1:3], limb_list[3:5], limb_list[5:7]]) :
        diff = abs(np.asarray(corr_limb_arr[0])-np.asarray(corr_limb_arr[1])) # difference between the two markers
        mean_corr = np.mean(corr_limb_arr, axis = 0) # NOT nanmean: if one is nan, the mean should be nan (discard that stride!)
        mean_corr[diff>0.05] = np.nan # if difference between the two markers big, at least one is poorly tracked
        [mean_limb_arr.append(m) for m in mean_corr]
        
    yremv = np.where(np.nanmin(arrayY_strides, axis = 0) < 200)[0]
    arrayY_strides[:, yremv] = np.zeros((arrayY_strides.shape[0], len(yremv))) + np.nan
    arrayX_strides[:, yremv] = np.zeros((arrayY_strides.shape[0], len(yremv))) + np.nan
    arrayX_strides2[:, yremv] = np.zeros((arrayY_strides.shape[0], len(yremv))) + np.nan
    
    diffX = np.diff(arrayX_strides, axis = 0)
    diffY = np.diff(arrayY_strides, axis = 0)
    dists = np.nansum(np.sqrt(diffX**2 + diffY**2), axis = 0) # nan columns become 0
    dists[dists == 0] =np.nan
    
    if os.path.exists(Path(outputDir)/"metadataPyRAT_processed.csv"):
        if 'rl' in arrayIDer[0,3]:
            df = pd.DataFrame(np.vstack((mouseID, expDate, stimFreq, headLVL, weights, ages, headHW, sexes,
                                          strideNum, frac0arr, frac1arr, frac2diagarr, frac2hmltarr, 
                                          frac2hmlgHINDarr, frac2hmlgFOREarr, frac3arr, frac4arr,
                                          strideFreq, strideLength, limb_list[0], limb_list[1],
                                          limb_list[2], limb_list[3], limb_list[4], limb_list[5], 
                                          limb_list[6], mean_limb_list[0], mean_limb_list[1], mean_limb_list[2],
                                          duty_lH_arr, duty_rH_arr, duty_lF_arr, duty_rF_arr, 
                                          speed_arr, accel_arr, snoutBody_arr, bodyAngleRange_arr)).T,
                              columns = ['mouseID', 'expDate', 'stimFreq', 'headLVL', 'weight','age', 'headHW',
                                          'sex','strideNum', 'frac0','frac1','frac2diag','frac2hmlt',
                                          'frac2hmlgHIND','frac2hmlgFORE','frac3','frac4','strideFreq','strideLength', 
                                          limb_list_str[0], limb_list_str[1], limb_list_str[2], limb_list_str[3], 
                                          limb_list_str[4], limb_list_str[5], limb_list_str[6], mean_limb_str[0],
                                          mean_limb_str[1], mean_limb_str[2], 'duty_lH', 'duty_rH', 'duty_lF', 
                                          'duty_rF', 'speed', 'acceleration', 'snoutBodyAngle', 
                                          'bodyAngleRange'])
        else:
            df = pd.DataFrame(np.vstack((mouseID, expDate, stimFreq, headLVL, weights, ages, sexes,strideNum, 
                                          frac0arr, frac1arr, frac2diagarr, frac2hmltarr, frac2hmlgHINDarr,
                                          frac2hmlgFOREarr, frac3arr, frac4arr, strideFreq, strideLength,
                                          limb_list[0], limb_list[1], limb_list[2], limb_list[3], limb_list[4], 
                                          limb_list[5], limb_list[6], mean_limb_list[0], mean_limb_list[1], 
                                          mean_limb_list[2],
                                          duty_lH_arr, duty_rH_arr, duty_lF_arr, duty_rF_arr, 
                                          speed_arr, accel_arr, snoutBody_arr, bodyAngleRange_arr)).T,
                              columns = ['mouseID', 'expDate', 'stimFreq', 'headLVL', 'weight','age',
                                          'sex','strideNum', 'frac0','frac1','frac2diag','frac2hmlt',
                                          'frac2hmlgHIND','frac2hmlgFORE','frac3','frac4','strideFreq','strideLength', 
                                          limb_list_str[0], limb_list_str[1], limb_list_str[2], limb_list_str[3], 
                                          limb_list_str[4], limb_list_str[5], limb_list_str[6], mean_limb_str[0],
                                          mean_limb_str[1], mean_limb_str[2], 'duty_lH', 'duty_rH', 'duty_lF', 
                                          'duty_rF', 'speed', 'acceleration', 'snoutBodyAngle', 
                                          'bodyAngleRange'])
    else:
        df = pd.DataFrame(np.vstack((mouseID, expDate, stimFreq, headLVL, strideNum, 
                                      frac0arr, frac1arr, frac2diagarr, frac2hmltarr, frac2hmlgHINDarr,
                                      frac2hmlgFOREarr, frac3arr, frac4arr, strideFreq, strideLength,
                                      limb_list[0], limb_list[1], limb_list[2], limb_list[3], limb_list[4], 
                                      limb_list[5], limb_list[6], mean_limb_list[0], mean_limb_list[1], 
                                      mean_limb_list[2],duty_lH_arr, duty_rH_arr, duty_lF_arr, duty_rF_arr, 
                                      speed_arr, accel_arr, snoutBody_arr, bodyAngleRange_arr)).T,
                          columns = ['mouseID', 'expDate', 'stimFreq', 'headLVL', 
                                     'strideNum', 'frac0','frac1','frac2diag','frac2hmlt',
                                      'frac2hmlgHIND','frac2hmlgFORE','frac3','frac4','strideFreq','strideLength', 
                                      limb_list_str[0], limb_list_str[1], limb_list_str[2], limb_list_str[3], 
                                      limb_list_str[4], limb_list_str[5], limb_list_str[6], mean_limb_str[0],
                                      mean_limb_str[1], mean_limb_str[2], 'duty_lH', 'duty_rH', 'duty_lF', 
                                      'duty_rF','speed', 'acceleration', 'snoutBodyAngle', 
                                      'bodyAngleRange'])
    df.to_csv(os.path.join(outputDir, f'{yyyymmdd}_strideParamsMerged{appdx}_{limb}.csv'))
    
    
def compute_duty_cycle_difference(yyyymmdd, 
                                  appdx = '',
                                  outputDir = Config.paths["passiveOpto_output_folder"]):
    filepath = os.path.join(outputDir, f"{yyyymmdd}_dutyCycles{appdx}.csv")

    df = pd.read_csv(filepath)

    pairs = ['Fore','Hind']
    
    from copy import deepcopy
    df_reduced = deepcopy(df[df['limb'] == 'lH'])
    df_reduced.reset_index(inplace=True) 
    df_reduced.drop(['dutyCycle', 'limb', 'index', 'Unnamed: 0'], axis = 1, inplace=True)
    
    for pair in pairs:
        df_reduced[f'{pair}_dutyCycle'] = np.nanmean(np.vstack((
                                df[df['limb'] == f'l{pair[0]}']['dutyCycle'],
                                df[df['limb'] == f'r{pair[0]}']['dutyCycle'])).T, 
                                axis = 1)
    # df_reduced['Diff_dutyCycle'] = df_reduced['Hind_dutyCycle']-df_reduced['Fore_dutyCycle']
    df_reduced['Diff_dutyCycle'] = np.asarray(df_reduced['Hind_dutyCycle']/df_reduced['Fore_dutyCycle'])
    df_reduced['Diff_dutyCycle'].replace(0, np.nan, inplace=True)
    df_reduced.to_csv(os.path.join(outputDir, f"{yyyymmdd}_dutyCyclesReduced.csv"))



