import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

param = 'headHW'
variable = 'CoMy_mean'
variable_str = 'CoMy'
title = 'anteroposterior CoS'
yyyymmdds = ['2023-11-06', '2021-10-26']
lbl_list = ['MSA-def', 'CTRL']

fig, ax = plt.subplots(1,2, figsize=(1.5, 1.2), sharey=True)
ylin_last = []; vals_DFs={}
for i, (yyyymmdd, clr, lbl) in enumerate(zip(
        yyyymmdds,
        [FigConfig.colour_config['homolateral'][2], FigConfig.colour_config['headbars']],
        lbl_list,
                                    )):
    
    df, _ = data_loader.load_processed_data(outputDir=Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="forceplateData", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = f'_{param}')
    headHW_df, _ = data_loader.load_processed_data(outputDir=Config.paths["forceplate_output_folder"], 
                                                   dataToLoad='forceplateHeadHW', 
                                                   yyyymmdd=yyyymmdd)
    
    if '2023' in yyyymmdd:
        df = df.droplevel(level=0,axis=1)
        headHW_df = headHW_df.droplevel(level=0,axis=1)
    
    mice = np.unique(df.columns.get_level_values(0))
    
    variability_df = pd.DataFrame(columns=['mouseID', 'headHW', 'var_within_trials', 'var_across_trials'])

    for m in mice:
        df_mouse = df.loc[:, m]
        rlevels = df_mouse.columns.get_level_values(0).unique()
        for rlvl in rlevels:
            within_trial = []; trials_means = []
            trials = df_mouse.loc[:, rlvl].columns.get_level_values(0).unique()
            for tr in trials:
                # compute CoMy
                comy = (df_mouse.loc[:, (rlvl, tr, 'rF')] +\
                        df_mouse.loc[:, (rlvl, tr, 'lF')] -\
                        df_mouse.loc[:, (rlvl, tr, 'rH')] -\
                        df_mouse.loc[:, (rlvl, tr, 'lH')])
                within_trial.append(np.std(comy))
                trials_means.append(np.mean(comy))
            headhw = headHW_df.loc[:, (m, rlvl)].values.mean()
            new_row = pd.DataFrame({
                'mouseID': [m],
                'headHW': [headhw],
                'var_across_trials': [np.std(trials_means)],
                'var_within_trials': [np.mean(within_trial)]
                })
            variability_df = pd.concat((variability_df, new_row), ignore_index=True)
        
    vals = variability_df.groupby(['mouseID'])[['var_within_trials', 'var_across_trials']].mean()*Config.forceplate_config['fore_hind_post_cm']/2 
    vals_DFs[lbl] = vals
    
    for xc, col in enumerate(vals.columns):        
        ax[xc].scatter(np.repeat(i, vals.shape[0]),
                   vals[col],
                   c = clr,
                   s = 5,
                   alpha=0.6)
        xintx = 0.25 if i==0 else -0.25
    
        ax[xc].boxplot(vals[col], 
                   positions = [i+xintx], 
                   medianprops = dict(color = clr, linewidth = 1, alpha = 0.8),
                    boxprops = dict(color = clr, linewidth = 1, alpha = 0.8), 
                    capprops = dict(color = clr, linewidth = 1, alpha = 0.8),
                    whiskerprops = dict(color = clr, linewidth = 1, alpha = 0.8), 
                    flierprops = dict(mec = clr, linewidth = 1, alpha = 0.8, ms=2))
    
        ax[xc].set_xticks([0,1], labels=['MSA','CTRL'])
        
        if i == 1:    
            # ADD STATS
            t, p = scipy.stats.ttest_ind(vals_DFs[lbl_list[0]][col], 
                                         vals_DFs[lbl_list[1]][col],
                                         nan_policy = 'omit') 
            
            print(f"{col} MSA vs CTRL: t = {t:.2f}, p = {p:.5f}")
            psum = (p < np.asarray(FigConfig.p_thresholds)).sum()
            ptext = psum * "*" if psum > 0 else "n.s."
            ax[xc].text(0.5, 0.58, ptext)       

# FORMAT THE PLOT
ax[0].set_ylabel("AP CoS standard\ndeviation (cm)")
ax[0].set_title("Within")
ax[1].set_title("Across") # but within head height
ax[0].set_ylim(0,0.6)
plt.tight_layout()

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplateEGR3_variability_within_across.svg'),
            transparent = True,
            dpi = 300)

