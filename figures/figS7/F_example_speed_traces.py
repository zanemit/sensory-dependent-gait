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


egr3path = r"C:\Users\MurrayLab\Documents\PassiveOptoTreadmill\2023-08-14_beltSpeedData_egr3.csv"
egr3_ctrl_path = r"C:\Users\MurrayLab\Documents\PassiveOptoTreadmill\2023-09-21_beltSpeedData_egr3ctrl.csv"

egr3 = pd.read_csv(egr3path, header = [0,1,2,3], index_col = 0)
egr3_ctrl = pd.read_csv(egr3_ctrl_path, header = [0,1,2,3], index_col = 0)

fig, ax = plt.subplots(1,1,figsize = (1.3,1.3)) 
freq = '50Hz'
time = np.linspace(-0.5, 5.5, egr3.shape[0])
ax.axvspan(ymax=0.99,xmin=0, xmax=5, facecolor = '#b8cfd1', edgecolor = None, alpha = 0.2)
ax.text(0,180,"50 Hz stimulation", color = '#b8cfd1', size=5, alpha =  1, ha='left')
ax.text(2.5,156,"Control", color = 'grey', size=5, ha='center')
ax.text(2.5,138,"MSA-deficient", color = FigConfig.colour_config['homolateral'][2], size=5, ha='center')

for i, (df, clr, lbl) in enumerate(zip(
                            [egr3_ctrl, egr3],
                            ['grey',FigConfig.colour_config['homolateral'][2]],
                            ['CTRL littermates', 'MSA-deficient']
                            )):
    
    if lbl == 'MSA-deficient':
        mice = Config.passiveOpto_config['egr3_mice']
    else:
        mice = Config.passiveOpto_config['egr3_ctrl_mice']
    
    speed_arr = np.empty((df.shape[0], len(mice)))*np.nan
    print(f"{lbl}: {len(mice)} mice")
    for im, m in enumerate(mice):
        df_sub = df.loc[:, (m, slice(None), freq)]
        speed_arr[:, im] = np.nanmean(df_sub, axis = 1)
    meanval = np.nanmean(speed_arr, axis = 1)
    sem = scipy.stats.sem(speed_arr, axis = 1, nan_policy = 'omit')  
    ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., speed_arr.shape[1]-1) 
    if lbl != 'MS-deficient':
        ax.fill_between(time,
                        meanval-ci,
                        meanval+ci,
                        facecolor = clr,
                        edgecolor = None,
                        alpha = 0.3)
    ax.plot(time, 
             meanval,
             linewidth = 1.5,
             color = clr,
             label = lbl)
    speed_peak = meanval.max()
    speed_peak_idx = np.argwhere(meanval==speed_peak)[0][0]
    print(f"Speed peak: {speed_peak:.0f} +- {ci[speed_peak_idx]:.0f} cm/s")
    
ax.set_ylim(-0.1,180)
ax.set_yticks([0,45,90,135,180])
ax.set_xticks(np.arange(6))
ax.set_ylabel("Speed (cm/s)")
ax.set_xlabel("Time (s)")
ax.set_title("Passive treadmill")
plt.tight_layout()
fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"example_egr3_speed_averages.svg",
            transparent = True,
            dpi = 300)

