import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt


sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler


yyyymmdd = '2021-10-26'
param = 'headHW'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                        dataToLoad='forceplateData_groupedby', 
                                        yyyymmdd = yyyymmdd, 
                                        appdx = param)
t_smr = np.linspace(0, Config.forceplate_config["trial_duration"], df.shape[0])

conditions = np.unique(df.columns.get_level_values(0))
cond_titles = [f"{c[:5]},{c.split(',')[-1][:5]}]" for c in conditions]

# MEANS WITH SHADED 95% CI
fig, ax = plt.subplots(1, conditions.shape[0], 
                       figsize=(conditions.shape[0]*1.4, 1.2), sharey=True)

for icon, (cond, cond_t) in enumerate(zip(conditions, cond_titles)):  # loop over levels/intervals
    # forelimb data of all mice
    data_fore = np.asarray(df.loc[:, (cond, slice(None), 'fore_weight_frac')])
    # same for hindlimb data
    data_hind = np.asarray(df.loc[:, (cond, slice(None), 'hind_weight_frac')])
    # same for headplate data
    data_head = np.asarray(df.loc[:, (cond, slice(None), 'headplate_weight_frac')])
    # means and sem
    mean_fore = np.nanmean(data_fore, axis=1)
    mean_hind = np.nanmean(data_hind, axis=1)
    mean_head = np.nanmean(data_head, axis=1)
    sem_fore = scipy.stats.sem(data_fore, axis=1, nan_policy='omit')
    sem_hind = scipy.stats.sem(data_hind, axis=1, nan_policy='omit')
    sem_head = scipy.stats.sem(data_head, axis=1, nan_policy='omit')
    ci_fore = sem_fore * \
        scipy.stats.t.ppf((1 + 0.95) / 2., data_fore.shape[1]-1)
    ci_hind = sem_hind * \
        scipy.stats.t.ppf((1 + 0.95) / 2., data_hind.shape[1]-1)
    ci_head = sem_head * \
        scipy.stats.t.ppf((1 + 0.95) / 2., data_head.shape[1]-1)
    t_mid = int(mean_fore.shape[0]/2)
    print(f"{cond_t}: {mean_fore[t_mid]:.2f} +- {ci_fore[t_mid]:.2f}, {mean_hind[t_mid]:.2f} +- {ci_fore[t_mid]:.2f}")
    ax[icon].fill_between(t_smr, mean_fore - np.asarray(ci_fore), mean_fore +
                           np.asarray(ci_fore), facecolor=FigConfig.colour_config["forelimbs"], alpha=0.3)
    ax[icon].fill_between(t_smr, mean_hind - np.asarray(ci_hind), mean_hind +
                           np.asarray(ci_hind), facecolor=FigConfig.colour_config["hindlimbs"], alpha=0.3)
    ax[icon].fill_between(t_smr, mean_head - np.asarray(ci_head), mean_head +
                           np.asarray(ci_head), facecolor=FigConfig.colour_config["headplate"], alpha=0.3)
    ax[icon].plot(t_smr, np.nanmean(data_fore, axis=1), linewidth=1,
                   color=FigConfig.colour_config["forelimbs"])
    ax[icon].plot(t_smr, np.nanmean(data_hind, axis=1), linewidth=1,
                   color=FigConfig.colour_config["hindlimbs"])
    ax[icon].plot(t_smr, np.nanmean(data_head, axis=1), linewidth=1,
                   color=FigConfig.colour_config["headplate"])
    ax[icon].set_xlabel('Time (s)')
    ax[icon].set_xticks([0,1,2,3,4,5])
    ax[icon].set_title(cond_t, size = 6)
    ax[0].set_ylim(-0.5, 1)
    ax[0].set_yticks([-0.5,0,0.5,1])
ax[0].set_ylabel('Weight distribution')
lgd = fig.legend([([FigConfig.colour_config["forelimbs"]],"solid"), 
             ([FigConfig.colour_config["hindlimbs"]],"solid"), 
             ([FigConfig.colour_config["headplate"]],"solid")],
             ["forelimbs", "hindlimbs", "headplate"], handler_map={tuple: AnyObjectHandler()}, 
             loc = 'upper center', bbox_to_anchor=(0.3,-0.4,0.5,0.2),
             mode="expand", borderaxespad=0, ncol = 3)

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], yyyymmdd + f'_weight_v_time_{param}_CI_allMice.svg'), bbox_extra_artists = (lgd, ), bbox_inches = 'tight')
