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

df, headplate_df, df_v = forceplate_data_manager.weight_calibrate_dataframe(df, weightCalib, metadata_df)
fig_cols = []
limbs = ['rF', 'rH', 'lF', 'lH']

mouse = 'FAA1034572'
rl = 'rl7'
df_sub = df_v[mouse][rl]

# MEANS WITH SHADED 95% CI
fig, ax = plt.subplots(1, 1, figsize=(1.2, 1.2))
color_dict = {'rF': ('diagonal',2),
              'rH': ('homologous',2),
              'lF': ('homolateral',2),
              'lH': ('greys',0)}

for limb in limbs:
    limbdata = df_sub.loc[:, (slice(None), limb)]
    m = np.nanmean(limbdata, axis = 1)
    sem = scipy.stats.sem(limbdata, axis = 1, nan_policy = 'omit') 
    ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., limbdata.shape[0]-1)   
    fig_cols.append(FigConfig.colour_config[color_dict[limb][0]][color_dict[limb][1]])
    
    ax.fill_between(t_smr,
                    m - ci,
                    m + ci,
                    facecolor = FigConfig.colour_config[color_dict[limb][0]][color_dict[limb][1]],
                    edgecolor = None,
                    alpha=0.2)
    ax.plot(t_smr, 
            m, 
            color = FigConfig.colour_config[color_dict[limb][0]][color_dict[limb][1]],
            lw = 2)

ax.set_xlabel('Time (s)')
ax.set_xticks([0,1,2,3,4,5])
ax.set_ylim(0,0.5)
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
ax.set_ylabel('Fraction of body weight')
lgd = fig.legend([([fig_cols[0]],"solid"), 
              ([fig_cols[1]],"solid"), 
              ([fig_cols[2]],"solid"),
              ([fig_cols[3]],"solid")],
              limbs, handler_map={tuple: AnyObjectHandler()}, 
              loc = 'upper center', bbox_to_anchor=(0.15,-0.35,0.75,0.2),
              mode="expand", borderaxespad=0, ncol = 2)

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'MS2_{yyyymmdd}_weight_v_time_{param}_CI_allMice.svg'), bbox_extra_artists = (lgd, ), bbox_inches = 'tight')
