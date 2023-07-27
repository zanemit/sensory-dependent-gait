import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt


sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

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

df_v, headplate_df = forceplate_data_manager.weight_calibrate_dataframe(df, weightCalib, metadata_df)
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

# for icon, (cond, cond_t) in enumerate(zip(conditions, cond_titles)):  # loop over levels/intervals
#     # forelimb data of all mice
#     data_fore = np.asarray(df.loc[:, (cond, slice(None), 'fore_weight_frac')])
#     # same for hindlimb data
#     data_hind = np.asarray(df.loc[:, (cond, slice(None), 'hind_weight_frac')])
#     # same for headplate data
#     data_head = np.asarray(df.loc[:, (cond, slice(None), 'headplate_weight_frac')])
#     # means and sem
#     mean_fore = np.nanmean(data_fore, axis=1)
#     mean_hind = np.nanmean(data_hind, axis=1)
#     mean_head = np.nanmean(data_head, axis=1)
#     sem_fore = scipy.stats.sem(data_fore, axis=1, nan_policy='omit')
#     sem_hind = scipy.stats.sem(data_hind, axis=1, nan_policy='omit')
#     sem_head = scipy.stats.sem(data_head, axis=1, nan_policy='omit')
#     ci_fore = sem_fore * \
#         scipy.stats.t.ppf((1 + 0.95) / 2., data_fore.shape[1]-1)
#     ci_hind = sem_hind * \
#         scipy.stats.t.ppf((1 + 0.95) / 2., data_hind.shape[1]-1)
#     ci_head = sem_head * \
#         scipy.stats.t.ppf((1 + 0.95) / 2., data_head.shape[1]-1)
#     t_mid = int(mean_fore.shape[0]/2)
#     print(f"{cond_t}: {mean_fore[t_mid]:.2f} +- {ci_fore[t_mid]:.2f}, {mean_hind[t_mid]:.2f} +- {ci_fore[t_mid]:.2f}")
#     ax[icon].fill_between(t_smr, mean_fore - np.asarray(ci_fore), mean_fore +
#                            np.asarray(ci_fore), facecolor=FigConfig.colour_config["forelimbs"], alpha=0.3)
#     ax[icon].fill_between(t_smr, mean_hind - np.asarray(ci_hind), mean_hind +
#                            np.asarray(ci_hind), facecolor=FigConfig.colour_config["hindlimbs"], alpha=0.3)
#     ax[icon].fill_between(t_smr, mean_head - np.asarray(ci_head), mean_head +
#                            np.asarray(ci_head), facecolor=FigConfig.colour_config["headplate"], alpha=0.3)
#     ax[icon].plot(t_smr, np.nanmean(data_fore, axis=1), linewidth=1,
#                    color=FigConfig.colour_config["forelimbs"])
#     ax[icon].plot(t_smr, np.nanmean(data_hind, axis=1), linewidth=1,
#                    color=FigConfig.colour_config["hindlimbs"])
#     ax[icon].plot(t_smr, np.nanmean(data_head, axis=1), linewidth=1,
#                    color=FigConfig.colour_config["headplate"])
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
