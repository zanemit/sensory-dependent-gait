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
param = 'snoutBodyAngle'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = f'_{param}')

mice = np.unique(df['mouse'])
param_num = 5
minmaxs = ([],[])

fig, ax = plt.subplots(1, 1, figsize=(1.55, 1.5))
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    param_split = np.linspace(df_sub['param'].min()-0.0001, df_sub['param'].max(), param_num+1)
    xvals = [np.mean((a,b,)) for a,b in zip(param_split[:-1], param_split[1:])]
    df_grouped = df_sub.groupby(pd.cut(df_sub['param'], param_split)) 
    group_row_ids = df_grouped.groups
    
    minmaxs[0].append(np.nanmin(xvals))
    minmaxs[1].append(np.nanmax(xvals))
    
    for limb_str, limb_clr in zip(['hind_weight_frac', 'fore_weight_frac', 'headplate_weight_frac'],
                                  ['hindlimbs', 'forelimbs', 'headbars']):
        yvals = [np.mean(df_sub.loc[val,limb_str].values) for key,val in group_row_ids.items()]
        ax.plot(xvals, 
                 yvals, 
                 color=FigConfig.colour_config[limb_clr],  
                 alpha=0.4, 
                 linewidth = 0.5)
             
# fore-hind and comxy plot means
for i, (variable, variable_str, clr) in enumerate(zip(['fore_weight_frac', 'hind_weight_frac', 'headplate_weight_frac'],
                                                      ['foreWfrac', 'hindWfrac', 'headWfrac'],
                                                      ['forelimbs', 'hindlimbs', 'headbars'])):
    modLIN = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_linear_{variable_str}_{param}.csv")
    print(f"{variable} is modulated by {(modLIN['Estimate'][1]*100):.1f} ± {(modLIN['Std. Error'][1]*100):.1f} %/deg")
    
    x_pred = np.linspace(np.nanmin(minmaxs[0])-np.nanmean(df['param']), np.nanmax(minmaxs[1])-np.nanmean(df['param']), endpoint=True)
    ax.set_xlim(np.nanmin(minmaxs[0])-(0.1*(np.nanmax(minmaxs[1])-np.nanmin(minmaxs[0]))),
                np.nanmax(minmaxs[1])+(0.1*(np.nanmax(minmaxs[1])-np.nanmin(minmaxs[0]))))

    y_predLIN = modLIN['Estimate'][0] + modLIN['Estimate'][1] * x_pred + np.nanmean(df[variable])
    x_pred += np.nanmean(df['param'])
    ax.plot(x_pred, y_predLIN, linewidth=1, color=FigConfig.colour_config[clr]) 
    p_text = clr + ' ' + ('*' * (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
    if (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
        p_text += "n.s."
    ax.text(155,1.4-(0.18*i), p_text, ha = 'center', color = FigConfig.colour_config[clr])
        
ax.set_ylabel("Weight fraction")
ax.set_xlabel('Snout-hump angle (deg)')

ax.set_xticks([135,145,155,165,175])
ax.set_xlim(135,175)
ax.set_yticks([-0.5,0,0.5,1.0, 1.5])
ax.set_title("Head height trials")

plt.tight_layout(w_pad = 0, pad = 0, h_pad = 0)
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], yyyymmdd + f'_foreHindHeadWeights_{param}.svg'))
