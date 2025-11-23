import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import xarray as xr
import os
import sys
from matplotlib import pyplot as plt

np.seed=42

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_processing, treadmill_data_manager
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

outputDir = Config.paths['passiveOpto_output_folder']
yyyymmdd = '2022-08-18'
appdx = ''
df, _, _ = data_loader.load_processed_data(outputDir = outputDir, 
                                            dataToLoad = "strideParamsMerged", 
                                            yyyymmdd = '2022-08-18',
                                            appdx = '',
                                            limb = 'lH1')


passiveOptoData, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'passiveOptoData', yyyymmdd = yyyymmdd, appdx = appdx)
speedData, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'beltSpeedData', yyyymmdd = yyyymmdd, appdx = appdx)
bodyAngleData, _ = data_loader.load_processed_data(outputDir, dataToLoad = 'bodyAngles', yyyymmdd = yyyymmdd, appdx = appdx)

sba_range = [141, 149]
# sba_range = [166, 174]
mice=np.intersect1d(Config.passiveOpto_config['mice'],Config.injection_config['left_inj_imp'])

mouse = Config.passiveOpto_config['mice'][10] #8 (7,10,11,9)

stimfreq_dict = {
    '10Hz': '40ms', '20Hz': '20ms', '30Hz': '13.2ms', '40Hz': '9.999999ms', '50Hz': '8ms'
    }
# limb_str = 'lH1'
limb_dict = {'rF1': 'diagonal', 'lF1': 'homolateral', 'rH1': 'homologous', 'lH1': 'reference'}
ref_str = 'lH1'
nonref_strs = ['rH1', 'lF1', 'rF1']

df_sub = df.loc[
    # (df['mouseID'].isin(mice))&
    (df['mouseID']==mouse)&
    (df['snoutBodyAngle']>sba_range[0])&(df['snoutBodyAngle']<=sba_range[1]),
    : ]

unique_exps = df_sub[['expDate', 'mouseID', 'stimFreq', 'headLVL']].drop_duplicates()

t_on = 200
t_off = t_on + (400*5)
time_steps = 50

def resample_snippet(snippet, target_len):
    """
    resamples a 1 D array snippet to target_len using linear interpolation
    """
    old_x = np.linspace(0, 1, len(snippet))
    new_x = np.linspace(0, 1, target_len)
    return np.interp(new_x, old_x, snippet)

da_3d = xr.DataArray(
    np.empty((time_steps ,  0, len(limb_dict))),
    dims=("time", "exp", "limb"),
    coords={
        "time": np.arange(time_steps),
        "exp": [],
        "limb": list(limb_dict.keys())
        }
    )

for i_exp, exp_row in unique_exps.iterrows():
    print(f"Processing {da_3d.shape[1]}: {exp_row['expDate']}, {exp_row['mouseID']}, {exp_row['stimFreq']}, {exp_row['headLVL']}...")
  
    # process ref limb
    ref_limb = passiveOptoData.loc[:, (exp_row['mouseID'], str(exp_row['expDate']), exp_row['stimFreq'], exp_row['headLVL'], ref_str, 'x')]
    limb_filtered = utils_processing.butter_filter(ref_limb, filter_freq = 2)   
    # plt.plot(limb_filtered)    
    peaks_filt = find_peaks(limb_filtered, prominence = 50)[0]
    troughs_filt = find_peaks(limb_filtered*-1, prominence = 50)[0]
    peaks_filt, troughs_filt = utils_processing.are_arrays_alternating(peaks_filt, troughs_filt)
    peaks_true = []
    troughs_true = []
    [peaks_true.append(np.argmax(ref_limb[(pf-13):(pf+13)])+pf-13) for pf in peaks_filt if (pf >= 13 and pf+13 < len(ref_limb))]
    for pt in troughs_filt:
        if (pt >= 13 and pt+13 < len(ref_limb)):
            troughs_true.append(np.argmin(ref_limb[(pt-13):(pt+13)])+pt-13)
        elif pt >= 13:
            troughs_true.append(np.argmin(ref_limb[(pt-13):(len(ref_limb)-1)])+pt-13)
        else:
            troughs_true.append(np.argmin(ref_limb[:(pt+13)])) 
    peaks_true = np.asarray(peaks_true)
    troughs_true = np.asarray(troughs_true)
    troughs_in_view = troughs_true[(troughs_true>t_on)&(troughs_true<t_off)]
    stepStart = troughs_in_view[2]-t_on
    stepEnd = troughs_in_view[3]-t_on
    
    troughs_in_view = troughs_true[(troughs_true>t_on)&(troughs_true<t_off)]
    
    arr = pd.DataFrame(np.empty((t_off-t_on+400, 4))*np.nan,
                       columns = limb_dict.keys())
    for i_trough in range(len(troughs_in_view)-1):
        # check if SBA and speed requirements are met
        speed = np.nanmean(
                    speedData.loc[troughs_in_view[i_trough]:troughs_in_view[i_trough+1],
                                  (exp_row['mouseID'], str(exp_row['expDate']), exp_row['stimFreq'], exp_row['headLVL'])]
                    )
        sba = np.nanmean(
                    bodyAngleData.loc[troughs_in_view[i_trough]:troughs_in_view[i_trough+1],
                                  (exp_row['mouseID'], str(exp_row['expDate']), exp_row['stimFreq'], exp_row['headLVL'], 'snoutBody')]
                    )
        
        # print(speed)
        # if not ((speed>speed_range[0])and(speed<=speed_range[1])) or not ((sba>sba_range[0])and(sba<=sba_range[1])):
        if not ((sba>sba_range[0])and(sba<=sba_range[1])) and not (speed>5):
            continue
        
        da_new = xr.DataArray(
            np.full((time_steps, 1, len([ref_str] + nonref_strs)), np.nan),
            dims=("time", "exp", "limb"),
            coords={
                "time": np.arange(time_steps),
                "exp": [f"{exp_row['expDate']}_{exp_row['mouseID']}_{exp_row['stimFreq']}_{exp_row['headLVL']}_stride{i_trough}"],
                "limb": [ref_str] + nonref_strs
                }
            )
        for i, limb_str in enumerate([ref_str] + nonref_strs): # it is important for ref limb to be the first one
            limb = passiveOptoData.loc[:, (exp_row['mouseID'], str(exp_row['expDate']), exp_row['stimFreq'], exp_row['headLVL'], limb_str, 'x')]
            limb_scaled = limb / Config.passiveOpto_config["px_per_cm"][limb_str]
            limb_centred = limb_scaled-np.mean(limb_scaled)
            # limb_centred = limb_scaled-limb_scaled[0]
            arr.loc[:,limb_str] = limb_centred.values
              
            resampled = resample_snippet(
                arr.loc[troughs_in_view[i_trough]:troughs_in_view[i_trough+1], limb_str],
                time_steps
                )
            da_new.loc[:, :, limb_str] = resampled[:, None] - resampled[0]
           
        da_3d = xr.concat([da_3d, da_new], dim="exp")
            
    
fig, ax = plt.subplots(1,1, figsize=(1.3,3))
sample_tresh = 200
for i, (limb_str, clr, added) in enumerate(zip(
        [ref_str] + nonref_strs,
        ['greys', 'homologous', 'homolateral', 'diagonal'],
        [18, 13, 7, 0]
        )):
    clr_id = 1 if clr=='greys' else 2
    sample_size = sample_tresh if da_3d.coords['exp'].shape[0]>sample_tresh else da_3d.coords['exp'].shape[0]
    for trace in np.random.choice(da_3d.coords['exp'].values, sample_size, replace=False):
        ax.plot(da_3d.loc[:, trace, limb_str]+added, color=FigConfig.colour_config[clr][clr_id], alpha=0.08)
    ax.plot(np.nanmean(da_3d.loc[:, :, limb_str], axis=1)+added, color=FigConfig.colour_config[clr][clr_id], lw=2)
    ax.text(time_steps*0.95,added,limb_str.upper()[:-1], fontsize=6)
ax.set_xticks(np.linspace(0, time_steps, 5),labels=np.linspace(0,100,5).astype(int))
ax.set_xlabel('% through stride')
ax.set_title(f"({sba_range[0]}, {sba_range[1]}] deg")
plt.tight_layout()

# y-axis plots distance in centimetres!!

figtitle = f"limb_traces_{mouse}_{sba_range[0]}_{sba_range[1]}_deg.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)


