import sys
from pathlib import Path
import pandas as pd
import os
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_processing, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandlerDouble

import random

refLimb = 'lH1'
group_num = 4
param = 'snoutBodyAngle'
fig, ax = plt.subplots(3, 1, figsize=(1.7, 1.5), sharex = True)
bin_num = 20
xmin = 140
xmax = 180
ymax = 0.15
N_bootstrap = 1000
N_resample = 100
bootstrap_arr = np.empty((N_bootstrap, 3)) * np.nan

for axid, (cfg, yyyymmdd, clr, lnst, mice, appdx,lbl,a, quintile) in enumerate(zip(
        [Config.paths["mtTreadmill_output_folder"], Config.paths["passiveOpto_output_folder"], Config.paths["passiveOpto_output_folder"]],
        ['2021-10-23',  '2022-08-18',  '2022-08-18'],
        [FigConfig.colour_config['greys'][0],FigConfig.colour_config['homolateral'][0], FigConfig.colour_config['homolateral'][2]],
        ['solid', 'solid','solid'],
        [Config.mtTreadmill_config['mice_level'],  Config.passiveOpto_config['mice'], Config.passiveOpto_config['mice']],
        ['',  '', ''],
        ['Motorised: level trials', 'Passive: head low', 'Passive: head high'],
        [0.1,0.1,0.1],
        [None, 20, 80]
        )):   

    #TODO: should load the combined dataset (should save it from R first!) to have correct centred vals!
    df, _, _ = data_loader.load_processed_data(outputDir = cfg, 
                                                   dataToLoad="strideParams", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = appdx,
                                                   limb = 'lH1')
        
    histAcrossMice = np.empty((len(mice), bin_num)) * np.nan
    
    if cfg == Config.paths["passiveOpto_output_folder"]:
        # HH-specific computation
        hh_quintile = np.percentile(df['headHW'],quintile)
        df = df[df['headHW'] < hh_quintile]
    
    for im, m in enumerate(mice):
        df_sub = df[df['mouseID'] == m]
        
        values = df_sub[param][~np.isnan(df_sub[param])] 
            
        print(len(values))
        if values.shape[0] < (2*bin_num): # accept only mouse-condition subsets that contain >3sd entries
            print(f"Mouse {m} excluded because there are only {values.shape[0]} valid trials!")
            continue
                
        # histograms
        n, bins, patches = ax[axid].hist(values, 
                                         bin_num, 
                                         range = (xmin,xmax), 
                                         color = clr, 
                                         density = True, 
                                         alpha = a) #hatch='////', fill=True, edgecolor = Config.colour_config[d],
        bins_mean = bins[:-1] + np.diff(bins)
        histAcrossMice[im,:] = n
    histAcrossMice_mean = np.nanmean(histAcrossMice, axis = 0)    
    bins_points = np.concatenate(([bins[0]], np.repeat(bins[1:-1],2), [bins[-1]]))
    histo_points = np.repeat(histAcrossMice_mean,2)
    ax[axid].plot(bins_points, 
                  histo_points, 
                  color = clr, 
                  linewidth = 2,
                  linestyle = lnst)
    ax[axid].set_ylim(0,ymax)
    ax[axid].text(xmin+0.005*xmin, ymax, lbl, color = clr)
    ax[axid].set_yticks([0,0.15])
    
    # for ni in range(N_bootstrap):
    #     sample = np.empty(len(mice) * N_resample) * np.nan
    #     for im, m in enumerate(mice):
    #         df_sub = df[df['mouseID'] == m]
    #         values = df_sub[param][~np.isnan(df[param])] 
    #         if len(values) > N_resample:
    #             sample[(im*N_resample): (im*N_resample+N_resample)] = random.sample(list(values), N_resample)
    #     bootstrap_arr[ni, axid] = np.nanmean(sample)

# bootstrap_df = pd.DataFrame(bootstrap_arr, columns = ['motorised', 'passive-low', 'passive-high'])
        
ax[2].set_xlabel("Snout-hump angle (deg)")
ax[1].set_ylabel("Probability density")
plt.tight_layout()

# STATISTICS
import scipy.stats
import itertools
combs = list(itertools.combinations(np.arange(axid+1), 2))
n_c = len(combs)
# for c in combs:
#     t,p = scipy.stats.ttest_ind(bootstrap_df.iloc[:,c[0]], 
#                                 bootstrap_df.iloc[:,c[1]], 
#                                 nan_policy = 'omit')  
#     print(f"{bootstrap_df.columns[c[0]]} vs {bootstrap_df.columns[c[1]]}: t = {t:.2f}, p = {p:E}")
        
fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"passiveOpto_SBA_motorised_v_passive_HISTOGRAMS.svg",
            dpi =300)
            