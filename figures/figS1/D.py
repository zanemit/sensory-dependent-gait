import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig

yyyymmdd = '2021-10-26'
param = 'headHW'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_'+param)

mice = np.unique(df['mouse'])

fig, ax = plt.subplots(1, 1, figsize=(1.6, 1.6))
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    headHWs = np.unique(df_sub['param'])
    
    for limb_str, limb_clr in zip(['CoMy_mean', 'CoMx_mean'],
                                  ['main', 'neutral']):
        yvals = [np.nanmean(df_sub[df_sub['param']==h][limb_str]) for h in np.unique(df_sub['param'])]
        ax.plot(headHWs, 
                 yvals, 
                 color=FigConfig.colour_config[limb_clr],  
                 alpha=0.4, 
                 linewidth = 0.5)
             
# fore-hind and comxy plot means
for i, (variable, variable_str, clr, title) in enumerate(zip(['CoMy_mean', 'CoMx_mean'],
                                                      ['CoMy', 'CoMx'],
                                                      ['main', 'neutral'],
                                                      ['anteroposterior', 'mediolateral'])):
    modQDR = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_quadratic_{variable_str}_{param}.csv")
    
    x_centered = df['param'] - np.nanmean(df['param'])
    x_pred = np.linspace(np.nanmin(x_centered), np.nanmax(x_centered), endpoint=True)

    y_predQDR = modQDR['Estimate'][0] + modQDR['Estimate'][1] * x_pred + modQDR['Estimate'][2] * x_pred**2 + np.nanmean(df[variable])
    x_pred += np.nanmean(df['param'])
    ax.plot(x_pred, y_predQDR, linewidth=1, color=FigConfig.colour_config[clr]) 
    p_text = title + ' ' + ('*' * (modQDR['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
    if (modQDR['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
        p_text += "n.s."
    ax.text(0.6,1-(0.18*i), p_text, ha = 'center', color = FigConfig.colour_config[clr])
        
ax.set_ylabel("Centre of gravity (cm)")
ax.set_xlabel('Weight-adjusted head height')

ax.set_xticks([0,0.6,1.2])
ax.set_xlim(-0.1,1.2)
ax.set_yticks([-1.0,-0.5,0,0.5,1.0])
ax.set_ylim(-1.0,1.0)

plt.tight_layout(w_pad = 0, pad = 0, h_pad = 0)
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], yyyymmdd + f'_CoMxCoMy_{param}.svg'))


