import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_processing, treadmill_data_manager
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

param = 'speed'
strideparam = 'strideLength'
limbRef = 'lH1'
configs = {"passiveOpto": Config.passiveOpto_config, 
           "mtTreadmill": Config.mtTreadmill_config}
limb_dict = {'rF1': 'RIGHT FORE', 'rH1': 'RIGHT HIND', 'lF1': 'LEFT FORE', 'lH1': 'LEFT HIND'}

def get_trial_summaries(df):
    # count number of trials
    num_trials = df["strideNum"].value_counts().iloc[0]
    split_rows = np.where(df["strideNum"]==1)[0]
    
    # define a summary dataframe
    cols = ["mouseID", "expDate", "stimFreq", "headLVL", "weight", "age", 
            # "headHW",
            "strideNum","snoutBodyAngle", "speed", "rH1", "lF1",
            "rF1","snoutBodyAngle_std", "speed_std", "rH1_std", "lF1_std",
            "rF1_std"]
    num_metadata_cols = np.where(np.asarray(cols)=='strideNum')[0][0]
    summary_df = pd.DataFrame(np.empty((num_trials, len(cols)))*np.nan,
                              columns = cols)
    summary_df[["mouseID", "stimFreq", "headLVL"]] = summary_df[["mouseID", "stimFreq", "headLVL"]].astype(str) 
    
    for row_id in range(len(split_rows)):
        if row_id == len(split_rows)-1:
            df_sub = df.iloc[split_rows[row_id]:]
            stridenum = [len(split_rows)-split_rows[row_id]]
        else:
            stridenum = [split_rows[row_id+1]-split_rows[row_id]]
            df_sub = df.iloc[split_rows[row_id]:split_rows[row_id+1]]
                             
        trial_info = df_sub.loc[split_rows[row_id],cols[:num_metadata_cols]]  
        numerical_info = df_sub.loc[:,["snoutBodyAngle", "speed"]] 
        circular_cols = [c for c in cols if "1" in c and "std" not in c]
        circular_info = df_sub.loc[:, circular_cols]
        info_to_add = np.concatenate((
                    trial_info, 
                    stridenum,
                    np.nanmean(numerical_info, axis=0),
                    scipy.stats.circmean(circular_info, axis=0, high=0.5, low=-0.5, nan_policy='omit'),
                    np.nanstd(numerical_info, axis=0),
                    scipy.stats.circstd(circular_info, axis=0, high=0.5, low=-0.5, nan_policy='omit')
                                      ))
        summary_df.iloc[row_id, :] = info_to_add
    summary_df["expDate"] = summary_df["expDate"].astype(int)
    return summary_df

def get_trial_data(summary_df, limb_dict, row_of_interest, incline=False):
    trial = summary_df.iloc[row_of_interest, :]
    stim_dur = Config.passiveOpto_config["stim_dur_dict"][trial["stimFreq"][:2]]
    folder = 'passiveOptoTreadmill_incline' if incline else 'passiveOptoTreadmill_analysed'
        
    inpath = f'G:\\{folder}\\{trial["expDate"]}\\ZM_{trial["expDate"]}_{trial["mouseID"]}_{trial["stimFreq"]}_{stim_dur}ms_{trial["headLVL"]}_analog.bin'
    
    dlc_files = [f for f in os.listdir(f'G:\\{folder}\\{trial["expDate"]}') if trial["mouseID"] in f and trial["stimFreq"] in f and trial["headLVL"] in f]
    videoL_file = [f for f in dlc_files if "videoL" in f and f.endswith(".h5")][0]
    videoR_file = [f for f in dlc_files if "videoR" in f and f.endswith(".h5")][0]
    
    inpath2 = f'G:\\{folder}\\{trial["expDate"]}\\{videoL_file}'
    inpath3 = f'G:\\{folder}\\{trial["expDate"]}\\{videoR_file}'

    d = np.fromfile(inpath, dtype = np.double).reshape(-1,7)
    on_frame, off_frame, _, _ = treadmill_data_manager.get_opto_triggers(d)
    
    trackingsL, bodyparts = utils_processing.preprocess_dlc_data(inpath2, likelihood_thr = 0.95)
    trackingsR, bodyparts = utils_processing.preprocess_dlc_data(inpath3, likelihood_thr = 0.95)    
    
    peaks_dict = {}
    troughs_dict = {}
    limbX_dict = {}
    
    for i, limb_str in enumerate(limb_dict.keys()):
        if 'l' in limb_str:
            multiplier = -1
            trackings = trackingsL
        elif 'r' in limb_str:
            multiplier = 1
            trackings = trackingsR
        else:
            raise ValueError("Bad limb_str?")
        limb = (trackings[limb_str]['x']*multiplier)#[t_on:t_off] # this makes troughs swing onsets for all legs
    
        limb_filtered = utils_processing.butter_filter(limb, filter_freq = 2)   
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
        peaks_dict[limb_str] = np.asarray(peaks_true)
        troughs_dict[limb_str] = np.asarray(troughs_true)
        limbX_dict[limb_str] = limb
    return peaks_dict, troughs_dict, limbX_dict, on_frame

def get_xcoord_data(summary_df, df, limb_dict, row_of_interest, xlims, incline=False):
    trial = summary_df.iloc[row_of_interest, :]
    
    arr = np.empty((len(limb_dict),2000))*np.nan
    
    clr_dict = {'rF1': 'diagonal', 'rH1': 'homologous', 'lF1': 'homolateral', 'lH1': 'greys'}
    shift_dict = {'lH1': 11.5, 'lF1': -4.5, 'rH1': 4, 'rF1': -11}
    
    _, troughs_dict, limbXdict, on_frame = get_trial_data(summary_df, limb_dict, row_of_interest, incline=incline)

    for i, limb_str in enumerate(limb_dict.keys()):
        limb = limbXdict[limb_str][on_frame:on_frame+2000] / Config.passiveOpto_config["px_per_cm"][limb_str]
        limb = limb - np.mean(limb[:50])
        limb_filtered = utils_processing.butter_filter(limb, filter_freq = 7) + shift_dict[limb_str]
        arr[i, :] = limb_filtered
    
    # phases = df[(df["mouseID"]==trial["mouseID"]) & (df["expDate"]==trial["expDate"])&(df["stimFreq"]==trial["stimFreq"])&(df["headLVL"]==trial["headLVL"])]['lF1']*2*np.pi
    # phases[phases<0] = phases[phases<0]+(2*np.pi)
    # print(phases)
    indep_var = df[(df["mouseID"]==trial["mouseID"]) & (df["expDate"]==trial["expDate"])&(df["stimFreq"]==trial["stimFreq"])&(df["headLVL"]==trial["headLVL"])]['snoutBodyAngle'].values
    if incline:
        indep_var = df[(df["mouseID"]==trial["mouseID"]) & (df["expDate"]==trial["expDate"])&(df["stimFreq"]==trial["stimFreq"])&(df["headLVL"]==trial["headLVL"])]['headLVL']
        indep_var = np.asarray([-int(x[3:]) for x in indep_var])
    speeds = df[(df["mouseID"]==trial["mouseID"]) & (df["expDate"]==trial["expDate"])&(df["stimFreq"]==trial["stimFreq"])&(df["headLVL"]==trial["headLVL"])]['speed'].values
    
    fig, ax = plt.subplots(1, 1, figsize=(3,1.7))
    for i, key in enumerate(limb_dict.keys()):
        ax.plot(arr[i,:], 
                lw=1.5,
                color=FigConfig.colour_config[clr_dict[key]][2])
        ax.text(xlims[1], arr[i, :].mean(), key[:-1].upper(), color='black')
    
    xmin = xlims[0]; xmax = xlims[1]
    ax.set_xlim(xmin, xmax)
    xticks = np.arange(xmin, xmax, 50)
    xticklabels = np.linspace(0, (xticks[-1]-xticks[0])/Config.passiveOpto_config["fps"], len(xticks))
    ax.set_xticks(xticks, labels = xticklabels)
    
    speeds_list = []; indep_var_list = []
    num_troughs_before_on_frame = (np.asarray(troughs_dict["lH1"])<on_frame).sum()
    print(troughs_dict["lH1"])
    for i, t in enumerate(troughs_dict["lH1"]) :
        # ADD STRIDE LINES TO THE PLOT
        ax.axvline(t-on_frame, ymin = 0, ymax = 1, color = 'black', ls = 'dashed',zorder = -1)
        
        if (i != len(troughs_dict["lH1"])-1) and ((t-on_frame)<xmax) and ((t-on_frame)>xmin):
            for otherlimb in np.setdiff1d(list(limb_dict.keys()), ['lH1']):
                i_phase = i - num_troughs_before_on_frame
                # troughs_dict begins at 0 (t-on_frame makes 0 the on_frame)
                # df begins at on_frame-200 
                phases = df[(df["mouseID"]==trial["mouseID"]) & (df["expDate"]==trial["expDate"])&(df["stimFreq"]==trial["stimFreq"])&(df["headLVL"]==trial["headLVL"])][otherlimb]*2*np.pi
                phases[phases<0] = phases[phases<0]+(2*np.pi)
                # if i==0:
                #     print(otherlimb, '\n', phases/np.pi)
                #     print(t, t-on_frame)
                if i_phase<len(phases):
                    phase_string = f"{phases.iloc[i_phase]/np.pi:.1f}π"
                    phase_string = f"{phases.iloc[i_phase]/np.pi:.0f}π" if ".0" in phase_string else phase_string
                    phase_string = "π" if phase_string=="1π" else phase_string
                    phase_string = "0" if phase_string=="0π" else phase_string
                    xax_pos_shift = (troughs_dict["lH1"][i+1] - t)/2
                    if t-on_frame+xax_pos_shift <xmax:
                        ax.text(t-on_frame+xax_pos_shift, shift_dict[otherlimb]+2, phase_string, ha='center',
                                color=FigConfig.colour_config[clr_dict[otherlimb]][2], size=5.5)
                else:
                    continue
            if i<len(speeds):
                speeds_list.append(speeds[i_phase])
                indep_var_list.append(indep_var[i_phase])
    
    ax.set_ylim(-15,15)
    ax.set_yticks(np.arange(-10,10.5,5))#, labels = ["-3","","","0","","","3"])
    ax.text(xmin-55,14,"anterior", color='black', size=5)
    ax.text(xmin-55,-15.5,"posterior", color='black', size=5)
    ax.set_ylabel("Horizontal foot position (cm)")

    if incline:
        ornt = "incline" if "-" in summary_df.loc[:,'headLVL'].iloc[row_of_interest] else "decline"
        slope = f"{-int(summary_df.loc[:,'headLVL'].iloc[row_of_interest][3:])} deg"
        ax.set_title(f"{ornt} ({slope}), {summary_df.loc[:,'stimFreq'].iloc[row_of_interest]}, {summary_df.loc[:,'speed'].iloc[row_of_interest]:.0f} cm/s")
    else:
        # ornt = "upward" if summary_df.loc[:,'snoutBodyAngle'].iloc[row_of_interest] > 165 else "downward"
        sba = np.nanmean(indep_var_list)
        ornt = "upward" if sba > 165 else "downward"
        # ax.set_title(f"{ornt} ({summary_df.loc[:,'snoutBodyAngle'].iloc[row_of_interest]:.0f} deg), {summary_df.loc[:,'stimFreq'].iloc[row_of_interest]}, {summary_df.loc[:,'speed'].iloc[row_of_interest]:.0f} cm/s")
        ax.set_title(f"{ornt} ({sba:.0f} deg), {summary_df.loc[:,'stimFreq'].iloc[row_of_interest]}, {np.nanmean(speeds_list):.0f} cm/s")
    
    figtitle = f"single_trial_phases_row{row_of_interest}_xmin{xmin}_xmax{xmax}_{summary_df.loc[:,'stimFreq'].iloc[row_of_interest]}_{summary_df.loc[:,'headLVL'].iloc[row_of_interest]}.svg"
    plt.tight_layout()
    fig.savefig(Path(FigConfig.paths['savefig_folder'])/figtitle,
                dpi=300,
                transparent=True)
#%%
# HEAD HEIGHT
df, _, _ = data_loader.load_processed_data(outputDir = Config.paths[f"{list(configs.keys())[0]}_output_folder"], 
                                            dataToLoad = "strideParams", 
                                            yyyymmdd = '2022-08-18',
                                            appdx = "",
                                            limb = "lH1")
summary_df = get_trial_summaries(df)

low_hh = ['rl17', 'rl12']
low_rows = summary_df[summary_df['headLVL'].isin(low_hh)].index
get_xcoord_data(summary_df, df, row_of_interest=low_rows[0], xlims=(65,475), limb_dict={'lF1': 'LEFT FORE', 'lH1': 'LEFT HIND', 'rH1': 'RIGHT HIND', 'rF1': 'RIGHT FORE'})
#535,945
# thesis: row_of_interest=80, xlims=(137,547)
# row_of_interest = 82, xlims=(535,945) - this is good, except it is not 149 deg


high_hh = ['rl-8', 'rl-3']
high_rows = summary_df[summary_df['headLVL'].isin(high_hh)].index
get_xcoord_data(summary_df, df, row_of_interest=high_rows[77], xlims=(530,940), limb_dict={'lF1': 'LEFT FORE', 'lH1': 'LEFT HIND', 'rH1': 'RIGHT HIND', 'rF1': 'RIGHT FORE'})
# thesis: row of interest 707,xlims=(828,1238) #66

# DOWNWARD (151 deg), 56 cm/s, 30Hz, row_of_interest=82, xlims=(535,945) - homolateral around pi
# DOWNWARD (151 deg), 73 cm/s, 30Hz, row_of_interest=706, xlims=(500,910) - homolateral around 1.5pi
# DOWNWARD (160 deg), 91 cm/s, 30Hz, row_of_interest=705, xlims=(250,660) - homolateral around 1.5pi
# DOWNWARD (165 deg), 86 cm/s, 50Hz, row_of_interest=high_rows[67], xlims=(75,485)
# row_of_interest=high_rows[76] -- pi but UPWARD (168 deg)
# UPWARD (167 deg), 72 cm/s, 20Hz, row_of_interest=high_rows[77], xlims=(530,940)


#%%
# INCLINE
df, _, _ = data_loader.load_processed_data(outputDir = Config.paths[f"{list(configs.keys())[0]}_output_folder"], 
                                            dataToLoad = "strideParams", 
                                            yyyymmdd = '2022-08-18',
                                            appdx = "_incline",
                                            limb = "lH1")
summary_df = get_trial_summaries(df)

get_xcoord_data(summary_df, limb_dict={'lF1': 'LEFT FORE', 'lH1': 'LEFT HIND'}, row_of_interest=64, xlims=(87,497), incline=True)

get_xcoord_data(summary_df, limb_dict={'lF1': 'LEFT FORE', 'lH1': 'LEFT HIND'}, row_of_interest=1255, xlims=(137,547), incline=True)
