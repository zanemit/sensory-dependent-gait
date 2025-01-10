import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

yyyymmdd = '2022-04-04'
param = 'levels'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = f'_{param}')

mice = np.unique(df['mouse'])
fig, axes = plt.subplots(1, 1, figsize=(1.3, 1.39))

limb_clr = 'homolateral'
limb_str = 'hind_weight_frac'
tlt = 'Surface slope trials'
variable_str = 'hindWfrac'

starts = np.empty(len(mice)); ends = np.empty(len(mice))
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    headHWs = np.unique(df_sub['param'])
    
    yvals = [np.nanmean(df_sub[df_sub['param']==h][limb_str]) for h in np.unique(df_sub['param'])]
    starts[im] = yvals[0]; ends[im] = yvals[-1]
    axes.plot(headHWs, 
             yvals, 
             color=FigConfig.colour_config[limb_clr][2],  
             alpha=0.4, 
             linewidth = 0.7)
print(np.nanmean(df[limb_str]))            
# # fore-hind and comxy plot means
modLIN = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_linear_{variable_str}_{param}.csv")

x_centered = df['param'] - np.nanmean(df['param'])
x_pred = np.linspace(np.nanmin(x_centered), np.nanmax(x_centered), endpoint=True)

y_predLIN = modLIN['Estimate'][0] + modLIN['Estimate'][1]*x_pred + np.nanmean(df[limb_str])
x_pred += np.nanmean(df['param'])
axes.plot(x_pred, 
        y_predLIN, 
        linewidth=1.5, 
        color=FigConfig.colour_config[limb_clr][2], 
        linestyle = 'solid') 

# p values
print(f"p-value: {modLIN['Pr(>|t|)'][1]}")
p_text = "slope: " + ('*' * (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
if (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
axes.text(0,1.45, p_text, ha = 'center', 
        color = FigConfig.colour_config[limb_clr][2], 
        fontsize = 5)
        
axes.set_xlabel('Surface slope (deg)')

axes.set_xticks([-40,-20,0,20,40][::2])
axes.set_xlim(-50,50)
axes.set_yticks(np.array([0,0.5,1.0, 1.5]))
axes.set_ylim(-0.3,1.5)
axes.set_title(tlt)

diffs = ends-starts
print(f"Max decline: {np.mean(starts):.3f} ± {scipy.stats.sem(starts):.3f}")
print(f"Max incline: {np.mean(ends):.3f} ± {scipy.stats.sem(ends):.3f}")
print(f"Mean change over 80 degrees: {np.mean(diffs):.3f} ± {scipy.stats.sem(diffs):.3f}")

# if A is significant, it means that the data asymptotes at a non-zero value
# if B is significant, it means that there is a (monotonic?) change in y as the x changes
# if k is significant, it means that y approaches the asymptote in an exponential manner

        
axes.set_ylabel("Hindlimb weight fraction")
# axes[2].set_ylabel("Total detected weight (%)")
plt.tight_layout()
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_hindHeadWeights_{param}_{limb_str}.svg'),
            transparent = True,
            dpi = 300)
