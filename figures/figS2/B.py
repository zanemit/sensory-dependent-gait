import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import os

sys.path.append(r"C:\Users\MurrayLab\thesis")

import scipy.stats
from processing import data_loader, utils_processing, treadmill_data_manager
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

# PLOT DLC TRACKING PASSIVE OPTO
inpath = r'Z:\murray\Zane\Treadmill\Analysed\210909\ZM_210909_FAA1034570_50Hz_8ms_rl3_analog.bin'
inpath2 = r'Z:\murray\Zane\Treadmill\Analysed\210909\ZM_210909_FAA1034570_50Hz_8ms_rl3_videoLDLC_resnet101_PassiveOptoTreadmill2-LSep9shuffle1_650000.h5'
inpath3 = r'Z:\murray\Zane\Treadmill\Analysed\210909\ZM_210909_FAA1034570_50Hz_8ms_rl3_videoRDLC_resnet101_PassiveOptoTreadmill2-ROct5shuffle1_650000.h5'
d = np.fromfile(inpath, dtype = np.double).reshape(-1,7)
on_frame, off_frame, _, _ = treadmill_data_manager.get_opto_triggers(d)

trackingsL, bodyparts = utils_processing.preprocess_dlc_data(inpath2, likelihood_thr = 0.95)
trackingsR, bodyparts = utils_processing.preprocess_dlc_data(inpath3, likelihood_thr = 0.95)
limb_str = 'lH1'
dataLabelDict = {'lF1': 'homolateral', 'rF1': 'diagonal', 'rH1': 'homologous', 'lH1': 'greys'}

def plot_example_dlc(trackings, limb_dict, t_on):
    figA, axA = plt.subplots(4,1, figsize=(2.8,0.8), sharex = True)
    
    for i, limb_str in enumerate(limb_dict.keys()):
        if 'l' in limb_str:
            multiplier = -1
        elif 'r' in limb_str:
            multiplier = 1
        else:
            raise ValueError("Bad limb_str?")
        limb = trackings[limb_str]['x']*multiplier
        limb_filtered = utils_processing.butter_filter(limb, filter_freq = 2)   
        # plt.plot(limb_filtered)    
        peaks_filt = find_peaks(limb_filtered, prominence = 50)[0]
        troughs_filt = find_peaks(limb_filtered*-1, prominence = 50)[0]
        peaks_filt, troughs_filt = utils_processing.are_arrays_alternating(peaks_filt, troughs_filt)
        peaks_true = []
        troughs_true = []
        [peaks_true.append(np.argmax(limb[(pf-13):(pf+13)])+pf-13) for pf in peaks_filt if (pf >= 13 and pf+13 < len(limb))]
        for pt in troughs_filt:
            if (pt >= 13 and pt+13 < len(limb)):
                troughs_true.append(np.argmin(limb[(pt-13):(pt+13)])+pt-13)
            elif pt >= 13:
                troughs_true.append(np.argmin(limb[(pt-13):(len(limb)-1)])+pt-13)
            else:
                troughs_true.append(np.argmin(limb[:(pt+13)])) 
        peaks_true = np.asarray(peaks_true)
        troughs_true = np.asarray(troughs_true)
        fps = 400  
        x_labels = [-1,0,1,2,3,4,5,6]
        x_ticks = np.linspace(t_on-fps, t_on+(6*fps), len(x_labels), endpoint = True)
    
        axA[i].axvspan(t_on,t_on+(5*fps), color = '#daf6fd')
        limb = limb / Config.passiveOpto_config["px_per_cm"][limb_str]
        axA[i].plot(limb, color = FigConfig.colour_config[dataLabelDict [limb_str]][2], linewidth = 0.7) 
        axA[i].scatter(troughs_true, limb[troughs_true], color = '#ABABAB', zorder = 5, s=1)  
        axA[i].scatter(peaks_true, limb[peaks_true], color = 'black', zorder=5, s=1)  
        axA[i].set_xlim(t_on-fps,t_on+(6*fps))
        axA[i].set_ylim(np.nanmean(limb)-5, np.nanmean(limb)+5)
        axA[i].set_xticks(x_ticks)
        axA[i].set_xticklabels(x_labels)
    plt.tight_layout()
    
    words = inpath.split('_')
plot_example_dlc(trackingsL, dataLabelDict, t_on = 480)