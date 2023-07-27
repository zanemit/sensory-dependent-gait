import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

param = 'levels'
param_num = 5
legend_colours = []

fig, ax = plt.subplots(1, 1, figsize=(1.55,1.5))

yyyymmdd = '2022-04-04'
limb_str = 'headplate_weight_frac'
limb_clr = 'headbars'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = f"_{param}")
mice = np.unique(df['mouse'])

for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    levels = np.unique(df_sub['param'])
    print(np.corrcoef(df_sub['param'], df_sub[limb_str])[0,1])
    
    yvals = [np.nanmean(df_sub[df_sub['param']==lvl][limb_str]) for lvl in levels]
    ax.plot(levels, 
             yvals, 
             color=FigConfig.colour_config[limb_clr],  
             alpha=0.4, 
             linewidth = 1)
# legend_elements.append(Line2D([0], [0], color = FigConfig.colour_config['headplate'], linestyle = lnst, label = name, linewidth = 1))

yyyymmdd = '2022-04-04'      
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_levels')         
# fore-hind and comxy plot means
meta_p = []
variable_str = 'headWfrac'
# for k in range(2):
modQDR = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_quadratic_{variable_str}_{param}.csv")

x_centered = df['param'] - np.nanmean(df['param'])
x_pred = np.linspace(np.nanmin(x_centered), np.nanmax(x_centered), endpoint=True)

# if k == 0: # head height 5 is the default condition
y_predQDR = modQDR['Estimate'][0] + modQDR['Estimate'][1]*x_pred + modQDR['Estimate'][2]*x_pred**2 + np.nanmean(df[limb_str])
x_pred += np.nanmean(df['param'])
ax.plot(x_pred, 
        y_predQDR, 
        linewidth=2, 
        color=FigConfig.colour_config[limb_clr], 
        linestyle = 'solid') 

p_text = 'slope ' + ('*' * (modQDR['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
if (modQDR['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
ax.text(0,0.9, p_text, ha = 'center', color = FigConfig.colour_config[limb_clr])

# if k == 0:
legend_colours.append(FigConfig.colour_config[limb_clr])
meta_p.append((modQDR['Pr(>|t|)'][2] < FigConfig.p_thresholds).sum())
               
ax.set_ylabel("Headbar weight fraction")
ax.set_xlabel('Incline (deg)')

ax.set_xticks([-40,-20,0,20,40])
ax.set_xlim(-42,40)
ax.set_yticks([-2,-1,0,1])
ax.set_title("Slope trials")


plt.tight_layout(w_pad = 0, pad = 0, h_pad = 0)

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'MS3_{yyyymmdd}_x_foreHindHeadWeights_{param}.svg'),
            dpi=300)
