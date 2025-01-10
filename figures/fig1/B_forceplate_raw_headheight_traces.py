import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt


sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, forceplate_data_manager
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

mouse = 'FAA1034570'
rl = 'rl2'

yyyymmdd = '2021-10-26'
param = 'headHW'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                        dataToLoad='forceplateData', 
                                        yyyymmdd = yyyymmdd, 
                                        appdx = param)
# import weight-voltage calibration files
weightCalib, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                 dataToLoad = 'weightCalibration', 
                                                 yyyymmdd = yyyymmdd)
# import mouse weight metadata
metadata_df = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                              dataToLoad = 'metadataProcessed', 
                                              yyyymmdd = yyyymmdd)
    
t_smr = np.linspace(0, Config.forceplate_config["trial_duration"], df.shape[0])

df, headplate_df = forceplate_data_manager.weight_calibrate_dataframe(df, metadata_df, yyyymmdd = yyyymmdd)
limbs = ['rF', 'rH', 'lF', 'lH']

# MEANS WITH SHADED 95% CI
fig, ax = plt.subplots(1, 2, figsize=(1.1*2, 1.2), sharey = True)
for k, rl in enumerate(['rl17', 'rl2']):
    fig_cols = []; lnst_cols = []
    
    df_sub = df[mouse][rl]
    height = 'low' if k == 0 else 'high'
    
    for limb in limbs:
        clr = 'homologous' if 'F' in limb else 'homolateral'
        lnst = 'dashed' if 'r' in limb else 'solid'
        
        limbdata = df_sub.loc[:, (slice(None), limb)]
        m = np.nanmean(limbdata, axis = 1)
        sem = scipy.stats.sem(limbdata, axis = 1, nan_policy = 'omit') 
        ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., limbdata.shape[0]-1)   
        fig_cols.append(FigConfig.colour_config[clr][2])
        lnst_cols.append(lnst)
        
        ax[k].fill_between(t_smr,
                        m - ci,
                        m + ci,
                        facecolor = FigConfig.colour_config[clr][2],
                        edgecolor = None,
                        alpha=0.2)
        ax[k].plot(t_smr, 
                m, 
                color = FigConfig.colour_config[clr][2],
                ls = lnst,
                lw = 1.2)
        ax[k].set_xticks([0,1,2,3,4,5])
        ax[k].set_title(f"head {height}")
        ax[k].set_xlabel('Time (s)')

ax[0].set_ylim(0,0.5)
ax[0].set_yticks([0,0.1,0.2,0.3,0.4,0.5])
ax[0].set_ylabel('Weight fraction')
lgd = fig.legend([([fig_cols[0]],lnst_cols[0]), 
              ([fig_cols[1]],lnst_cols[1]), 
              ([fig_cols[2]],lnst_cols[2]),
              ([fig_cols[3]],lnst_cols[3])],
              [s.upper() for s in limbs], 
              handler_map={tuple: AnyObjectHandler()}, 
              loc = 'upper center', bbox_to_anchor=(0.2,-0.15,0.7,0.2),
              mode="expand", borderaxespad=0, ncol = 4)
plt.tight_layout(h_pad = 3)
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_raw_weight_v_time_{mouse}_rl{rl}.svg'), 
            transparent = True,
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight',
            dpi = 300)
