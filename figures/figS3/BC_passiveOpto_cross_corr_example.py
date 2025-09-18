import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import os

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

import scipy.stats
from processing import data_loader, utils_processing, treadmill_data_manager
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

# PLOT DLC TRACKING PASSIVE OPTO
inpath = r"G:\passiveOptoTreadmill_analysed\220302\ZM_220302_FAA1034836_50Hz_8ms_rl0_analog.bin"
inpath2 = r"G:\passiveOptoTreadmill_analysed\220302\ZM_220302_FAA1034836_50Hz_8ms_rl0_videoLDLC_resnet101_PassiveOptoTreadmill2-LSep9shuffle10_650000.h5"
inpath3 = r"G:\passiveOptoTreadmill_analysed\220302\ZM_220302_FAA1034836_50Hz_8ms_rl0_videoRDLC_resnet101_PassiveOptoTreadmill2-ROct5shuffle10_650000.h5"

d = np.fromfile(inpath, dtype = np.double).reshape(-1,7)
on_frame, off_frame, _, _ = treadmill_data_manager.get_opto_triggers(d)

trackingsL, bodyparts = utils_processing.preprocess_dlc_data(inpath2, likelihood_thr = 0.95)
trackingsR, bodyparts = utils_processing.preprocess_dlc_data(inpath3, likelihood_thr = 0.95)
limb_dict = {'rF1': 'diagonal', 'lF1': 'homolateral', 'rH1': 'homologous', 'lH1': 'reference'}
ref_str = 'lH1'
nonref_strs = ['rH1', 'lF1', 'rF1']
fps=Config.passiveOpto_config['fps']
t_on = on_frame-int(fps/4)#920
t_off = off_frame+int(fps/4)#1080

fig1, ax1 = plt.subplots(1, 1, figsize=(3.2,1.1))

arr = pd.DataFrame(np.empty((t_off-t_on, 4))*np.nan,
                   columns = limb_dict.keys())

for i, limb_str in enumerate(limb_dict.keys()):
    if 'l' in limb_str:
        multiplier = -1
        trackings = trackingsL
    elif 'r' in limb_str:
        multiplier = 1
        trackings = trackingsR
    else:
        raise ValueError("Bad limb_str?")
        
    limb = (trackings[limb_str]['x']*multiplier)[t_on:t_off]
    limb = limb / Config.passiveOpto_config["px_per_cm"][limb_str]
    limb_centred = limb-np.mean(limb)
    arr.loc[:,limb_str] = limb_centred
    
    ax1.axvspan(on_frame, off_frame, facecolor='lightblue', alpha=0.1, edgecolor=None)
    
    # PLOT TRACES
    ax1.plot(limb_centred + (10*i), 
            color = FigConfig.colour_config[limb_dict[limb_str]][2], 
            linewidth = 0.8) 
    # ax.scatter(peaks_true, limb[peaks_true], color = 'black', zorder=5, s=1)  
    ax1.set_xlim(t_on,t_off)
    ax1.set_xticks(np.arange(on_frame, off_frame+fps, fps), labels=np.arange(0,6))
    ax1.set_ylim(-5,35)
    ax1.set_yticks(np.arange(-5,35,5))

    if limb_str == ref_str:
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
        troughs_in_view = troughs_true[(troughs_true>t_on)&(troughs_true<t_off)]
                
        troughs_in_view = troughs_true[(troughs_true>t_on)&(troughs_true<t_off)]
        for i_trough in range(troughs_in_view.shape[0]-1) :
            # ADD STRIDE LINES TO THE PLOT
            homologue_str = 'r'+ ref_str[1:] if 'l' in ref_str else 'l'+ ref_str[1:]
            homologue  = (trackings['rH1']['x']*multiplier)[troughs_in_view[i_trough]:troughs_in_view[i_trough+1]]
            if abs(homologue.min()-homologue.max())>50:
                ax1.axvline(troughs_in_view[i_trough], ymin = 0, ymax = 1, color = 'black', ls = 'dashed',zorder = 5, lw=0.3)
                if i_trough == troughs_in_view.shape[0]-2:
                    ax1.axvline(troughs_in_view[i_trough+1], ymin = 0, ymax = 1, color = 'black', ls = 'dashed',zorder = 5, lw=0.3)

fig1.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'passive_treadmill_crosscorr_example_trace.svg'), 
            dpi =300,
            transparent = True)


fig2, ax2 = plt.subplots(3, 1, figsize=(1.5,1.5), sharex=True)

step_number = 6
stepStart = troughs_in_view[step_number]-t_on
stepEnd = troughs_in_view[step_number+1]-t_on
dlc_limb = limb_centred.values[stepStart:stepEnd]

# CORRELATION
n = dlc_limb.shape[0]
delays = np.linspace(-0.5*n, 0.5*n, n).astype(int)
for j, limb_str in enumerate(nonref_strs): # iterate over non-reference bps
    limb = arr[limb_str][t_on:t_off]
    
    corrs = []
    for t_delay in delays: # iterate over time delays
        try:
            dlc = limb[(stepStart+t_delay):(stepEnd+t_delay)]
            corr, _ = pearsonr(dlc, dlc_limb)
            corrs.append(corr)
        except:
            # print('Out of range!')
            corrs.append(np.nan)
    xvals = np.linspace(-0.5,0.5,len(corrs))
    ax2[j].set_xticks([-0.5,0,0.5],
                  labels = ["-0.5π", 0, "0.5π"])
    ax2[j].plot(xvals, 
            corrs,
            lw =1,
            color = FigConfig.colour_config[limb_dict[limb_str]][2])
    ax2[j].set_ylim(-1,1.05)
    # ax.set_x
    if not np.all(np.isnan(corrs)):
        max_corr = delays[np.nanargmax(corrs)] / delays.shape[0]
        ax2[j].axvline(max_corr, 0, 1, ls = 'solid', color = 'black')
        ax2[j].text(0.14,0.8,
                f"{2*max_corr:.1f}π") #appends a single value!

    # ax2[j].set_title(limb_dict[limb_str])

ax2[2].set_xlabel("Phase delay")
ax2[1].set_ylabel("Correlation coefficient")

plt.tight_layout()
fig2.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'passive_treadmill_crosscorr_example.svg'), 
            dpi =300,
            transparent = True)