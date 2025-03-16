import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, forceplate_data_manager
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

def get_stats_stars(pval, ncomp=1):
    ptxt = ('*' * (pval < np.asarray(FigConfig.p_thresholds)/ncomp).sum())
    if (pval < np.asarray(FigConfig.p_thresholds)/ncomp).sum() == 0:
        ptxt += "n.s."
    return ptxt

egr3path = r"C:\Users\MurrayLab\Documents\PassiveOptoTreadmill\2023-08-14_beltSpeedData_egr3.csv"
egr3_ctrl_path = r"C:\Users\MurrayLab\Documents\PassiveOptoTreadmill\2023-09-21_beltSpeedData_egr3ctrl.csv"

egr3 = pd.read_csv(egr3path, header = [0,1,2,3], index_col = 0)
egr3_ctrl = pd.read_csv(egr3_ctrl_path, header = [0,1,2,3], index_col = 0)

freq = '50Hz'
time = np.linspace(-0.5, 5.5, egr3.shape[0])

exp_dict = {}
custom_clrs = []
for i, (df, clr, lbl) in enumerate(zip(
                            [egr3_ctrl, egr3],#, vglut2],
                            ['grey',FigConfig.colour_config['homolateral'][2]],#, 'black'],
                            ['MSA control',  'MSA-deficient']#, 'vGlut2']
                            )):
   
    if lbl == 'MSA-deficient':
        mice = Config.passiveOpto_config['egr3_mice']
    elif lbl == 'MSA control':
        mice = Config.passiveOpto_config['egr3_ctrl_mice']
    else:
        mice = Config.passiveOpto_config['mice']
    
    mouse_auc_dict = {}
    for im, m in enumerate(mice):
        if m in df.columns.get_level_values(0).unique():
            if freq in df.loc[:, m].columns.get_level_values(1).unique():
                df_sub = df.loc[:, (m, slice(None), freq)]
                mouse_auc_dict[m] = []
                for col in range(df_sub.shape[1]):
                    mouse_auc_dict[m].append(np.trapz(df_sub.iloc[:,col], time))
    exp_dict[lbl] = [np.mean(vals) for key, vals in mouse_auc_dict.items()]  
    custom_clrs.append(clr)    

exp_df = pd.DataFrame([(key,value) for key,values in exp_dict.items() for value in values],
                      columns=['Exp', 'Distance'])

# PLOT BOXPLOTS
fig, ax = plt.subplots(1,1,figsize = (1.1,1.2)) 
boxplot_shift=-0.15
scatter_shift=0.15
for i, (vals, clr) in enumerate(zip(exp_dict.values(), custom_clrs)):
    ax.boxplot(vals, positions = [i+boxplot_shift], 
               medianprops = dict(color = clr, linewidth = 1, alpha = 0.6),
               boxprops = dict(color = clr, linewidth = 1, alpha = 0.6), 
               capprops = dict(color = clr, linewidth = 1, alpha = 0.6),
               whiskerprops = dict(color = clr, linewidth = 1, alpha = 0.6), 
               flierprops = dict(mec = clr, linewidth = 1, alpha = 0.6, ms=2))
    
    ax.scatter(np.repeat(i+scatter_shift, 
               len(vals)), 
               vals, 
               color =  clr, alpha = 0.8, s = 5, zorder = 3)
ax.set_xticks(np.arange(len(exp_dict.keys())), labels=['CTRL', 'MSA'])#, 'vGlut2'])
ax.set_ylim(-20,600)
ax.set_yticks(np.linspace(0,600,5))
ax.set_ylabel("Distance during\nstim (cm)")


# Egr3 comparison
ax.axhline(550,0.35,0.9, color='black')
_, p2 = scipy.stats.ttest_ind(exp_dict['MSA control'], exp_dict['MSA-deficient'])
ax.text(0.25,560,get_stats_stars(p2, ncomp=4))

plt.tight_layout()
fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"passiveOpto_distance_boxplots_exptypes.svg",
            transparent = True,
            dpi = 300)

