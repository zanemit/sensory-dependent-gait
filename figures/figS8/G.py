from pathlib import Path
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from processing import utils_processing, utils_math, mt_treadmill_data_manager
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def plot_figS8G():
    folder = Config.paths['mtTreadmill_output_folder']
    bodyparts = ['snout', 'ear']
    
    dlc_files = [f for f in os.listdir(folder) if f.endswith(".h5")]
    filename_stems = [f.split("_video")[0] for f in os.listdir(folder) if f.endswith(".avi")]
    
    fps = Config.mtTreadmill_config["fps"]
    max_trial_dur = Config.mtTreadmill_config["max_trial_duration"]
    
    dlc_dict = {} #np.empty((fps*max_trial_dur, 0 )) * np.nan
    for dlc in dlc_files:
        f_stem = dlc.split("_video")[0]
        analog_path = os.path.join(folder, f"{f_stem}_analog.bin")
        dlc_path = os.path.join(folder, dlc)
        
        analog_data = np.fromfile(analog_path, dtype = np.double).reshape(-1,7)
        dlc_data = pd.read_hdf(dlc_path)
        
        trialON, trialOFF, frame_num_per_trial, _, _, stimType = mt_treadmill_data_manager.get_trials_and_optotriggers(analog_data)
        
        tuples = []
        fr_start = 0;   fr_end = 0
        for trial, (on, off, fr) in enumerate(zip(trialON, trialOFF, frame_num_per_trial)):
            fr_end += fr
            treadmill_speed_trials = analog_data[on:off,2]
            treadmill_speed_trials_downsampled = utils_processing.downsample_data(treadmill_speed_trials, 
                                                                                  dlc_data.loc[fr_start:fr_end].shape[0]) # frames (in a trial) at which treadmill moves
            dlc_ear_lkhd_arr = np.asarray(dlc_data.loc[fr_start:fr_end, (slice(None), 'ear', 'likelihood')][treadmill_speed_trials_downsampled > 0.1])
            dlc_snout_lkhd_arr = np.asarray(dlc_data.loc[fr_start:fr_end, (slice(None), 'snout', 'likelihood')][treadmill_speed_trials_downsampled > 0.1])
            
            frac_bad_ear = (dlc_ear_lkhd_arr < 0.95).sum() / dlc_ear_lkhd_arr.shape[0]
            frac_bad_snout = (dlc_snout_lkhd_arr < 0.95).sum() / dlc_snout_lkhd_arr.shape[0]
            
            if frac_bad_ear < 0.1 and frac_bad_snout < 0.1:
                print(f"{f_stem} trial {trial} accepted! Continuing file processing...")
                
                _, expDate, mouseID, _, protocol = f_stem.split("_")
                
                for bp in bodyparts:
                    for coord in ['x', 'y', 'likelihood']:
                        arr = np.asarray(dlc_data.loc[fr_start:fr_end, (slice(None), bp, coord)][treadmill_speed_trials_downsampled > 0.1])
                        dlc_dict[(expDate, mouseID, protocol, trial, bp, coord)] = arr
            else:
                print(f"Tracking not good enough for {f_stem}! Bad frac snout: {frac_bad_snout:.2f}, ear: {frac_bad_ear:.2f}. Discarding...")
            
            fr_start += fr
    
    max_arr_len = max(len(v) for v in dlc_dict.values())
    padded_dlc_data = {k: np.pad(v.flatten(), (0, max_arr_len - len(v)), constant_values = np.nan) for k,v in dlc_dict.items()}
    dlc_df = pd.DataFrame.from_dict(padded_dlc_data, orient='index').T
    dlc_df.columns = pd.MultiIndex.from_tuples(dlc_df.columns)
    dlc_df.columns.names = ["expDate", "mouseID", "protocol", "trialNum", "bodypart", "coord"]
    
    unique_colnames = dlc_df.columns.droplevel([4,5]).unique()
    head_angle_dict = {}
    for ucol in unique_colnames:
        is_bad_ear = dlc_df.loc[:,(ucol[0],ucol[1],ucol[2],ucol[3],'ear','likelihood')]<0.95
        dlc_df.loc[is_bad_ear, (ucol[0],ucol[1],ucol[2],ucol[3],'ear',slice(None))] = np.nan
        
        is_bad_snout = dlc_df.loc[:,(ucol[0],ucol[1],ucol[2],ucol[3],'snout','likelihood')]<0.95
        dlc_df.loc[is_bad_snout, (ucol[0],ucol[1],ucol[2],ucol[3],'snout',slice(None))] = np.nan
        
        df_sub = dlc_df.loc[:,(ucol[0],ucol[1],ucol[2],ucol[3])].dropna(axis=0)
        angles = utils_math.angle_with_x(
                            df_sub.loc[:,('ear', 'x')], # earX, 
                            df_sub.loc[:,('ear', 'y')], # earY
                            df_sub.loc[:,('snout', 'x')], # snoutX 
                            df_sub.loc[:,('snout', 'y')], # snoutY, 
                                )
        head_angle_dict[ucol] = angles
    
    max_arr_len = max(len(v) for v in head_angle_dict.values())
    padded_angle_data = {k: np.pad(v.flatten(), (0, max_arr_len - len(v)), constant_values = np.nan) for k,v in head_angle_dict.items()}
    
    head_angle_arr = np.stack(list(padded_angle_data.values()), axis=0).T
    angles = -(head_angle_arr.flatten() - 10.75)
    print(f"Angle range: [{np.nanmin(angles):.0f},{np.nanmax(angles):.0f}], mean: {np.nanmean(angles):.0f}")
    
    import seaborn as sns
    fig, ax = plt.subplots(1,1,figsize=(1,1.2))
    sns.violinplot(angles,ax=ax, color=FigConfig.colour_config["homolateral"][2])
    ax.set_ylabel("Relative pitch angle\n(deg)")
    ax.set_ylim(-100,20)
    ax.axhline(0,-1,1, color='black',ls='dashed')
    
    plt.tight_layout()
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    savepath = Path(FigConfig.paths['savefig_folder']) / f"head_tilt_angle_mtTrdm.svg"
    fig.savefig(savepath,
                dpi =300,
                transparent=True)
    print(f"FIGURE SAVED AT {savepath}")
if __name__=="__main__":
    plot_figS8G() 
                
    
    
