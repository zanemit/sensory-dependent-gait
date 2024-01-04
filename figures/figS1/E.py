import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

yyyymmdd = '2021-10-26'
param = 'headHW'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_'+param)

mice = np.unique(df['mouse'])
variable = 'CoMy_mean'
clr = 'main'

fig, ax = plt.subplots(1, 1, figsize=(1.55, 1.5))
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    headHWs = np.unique(df_sub['param'])
    
    yvals = [np.nanmean(df_sub[df_sub['param']==h][variable]) for h in np.unique(df_sub['param'])]
    ax.plot(headHWs, 
             yvals, 
             color=FigConfig.colour_config[clr],  
             alpha=0.4, 
             linewidth = 1)
             
# fore-hind and comxy plot means
variable_str = 'CoMy'
title = 'anteroposterior CoS'

modQDR = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_quadratic_{variable_str}_{param}.csv")

x_centered = df['param'] - np.nanmean(df['param'])
x_pred = np.linspace(np.nanmin(x_centered), np.nanmax(x_centered), endpoint=True)

y_predQDR = modQDR['Estimate'][0] + modQDR['Estimate'][1] * x_pred + modQDR['Estimate'][2] * x_pred**2 + np.nanmean(df[variable])
x_pred += np.nanmean(df['param'])
ax.plot(x_pred, 
        y_predQDR, 
        linewidth=2, 
        color=FigConfig.colour_config[clr]) 
p_text = title + ' ' + ('*' * (modQDR['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
if (modQDR['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
ax.text(0.6,0.9, p_text, ha = 'center', color = FigConfig.colour_config[clr])
        
ax.set_ylabel("Centre of support (cm)")
ax.set_xlabel('Weight-adjusted head height')

ax.set_xticks([0,0.6,1.2])
ax.set_xlim(-0.1,1.2)
ax.set_yticks([-1.0,-0.5,0,0.5,1.0])
ax.set_ylim(-1.0,1.0)
ax.set_title("Head height trials")

plt.tight_layout(w_pad = 0, pad = 0, h_pad = 0)
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'MS2_{yyyymmdd}_CoMy_{param}.svg'))


