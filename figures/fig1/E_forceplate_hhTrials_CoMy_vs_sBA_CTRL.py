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

yyyymmdd = '2021-10-26'
param = 'snoutBodyAngle'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = f'_{param}')

mice = np.unique(df['mouse'])
param_num = 5
minmaxs = ([],[])

fig, axes = plt.subplots(1,1, figsize=(1.45, 1.47))

variable = 'CoMy_mean'
clr = 'homolateral'
variable_str = 'CoMy'
tlt = 'Head height trials'

    
diffs = np.empty(len(mice))
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    param_split = np.linspace(df_sub['param'].min()-0.0001, df_sub['param'].max(), param_num+1)
    xvals = [np.mean((a,b,)) for a,b in zip(param_split[:-1], param_split[1:])]
    df_grouped = df_sub.groupby(pd.cut(df_sub['param'], param_split)) 
    group_row_ids = df_grouped.groups
    
    minmaxs[0].append(np.nanmin(xvals))
    minmaxs[1].append(np.nanmax(xvals))
    
    yvals = [np.mean(df_sub.loc[val,variable].values)*Config.forceplate_config['fore_hind_post_cm']/2 for key,val in group_row_ids.items()]
    diffs[im] = yvals[-1]-yvals[0]
    axes.plot(xvals, 
             yvals, 
             color=FigConfig.colour_config[clr][2],  
             alpha=0.4, 
             linewidth = 0.7)

print(f"Mean change over 80 degrees: {np.mean(diffs):.3f} ± {scipy.stats.sem(diffs):.3f}")

modLIN = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_linear_BEST_{variable_str}_{param}.csv")
print(f"{variable} is modulated by {(modLIN['Estimate'][1]*Config.forceplate_config['fore_hind_post_cm']*100/2):.1f} ± {(modLIN['Std. Error'][1]*Config.forceplate_config['fore_hind_post_cm']*100/2):.1f} mm/deg")

x_pred = np.linspace(np.nanmin(minmaxs[0])-np.nanmean(df['param']), np.nanmax(minmaxs[1])-np.nanmean(df['param']), endpoint=True)
# print(np.nanmean(df[variable]))
y_predLIN = (modLIN['Estimate'][0] + modLIN['Estimate'][1] * x_pred + np.nanmean(df[variable])) * Config.forceplate_config['fore_hind_post_cm']/2
x_pred += np.nanmean(df['param'])
axes.plot(x_pred, 
        y_predLIN, 
        linewidth=1.5, 
        color=FigConfig.colour_config[clr][2]) 
p_text = 'slope: '+ ('*' * (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
if (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
axes.text(155,0.9, p_text, ha = 'center', color = FigConfig.colour_config[clr][2], fontsize = 5)

axes.set_xlabel('Snout-hump angle\n(deg)')
axes.set_xticks([135,145,155,165,175][::2])
axes.set_xlim(135,175)
axes.set_yticks([-1.0,-0.5,0,0.5,1.0])
axes.set_ylim(-1.3,1.0)
axes.set_title(tlt)

axes.set_ylabel("Anteroposterior\nCoS (cm)")
plt.tight_layout()
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_CoMy_vs_sBA.svg'),
            transparent = True,
            dpi =300)


