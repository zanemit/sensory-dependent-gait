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
from figures.fig_config import AnyObjectHandler

param = 'levels'
param_num = 5
legend_colours = []
yyyymmdd = '2022-04-04'


fig, ax = plt.subplots(1, 1, figsize=(1.55, 1.5))
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = f"_{param}")
mice = np.unique(df['mouse'])

variable = 'CoMy_mean'
clr = 'homolateral'

for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    levels = np.unique(df_sub['param'])
    
    yvals = [np.nanmean(df_sub[df_sub['param']==lvl][variable])*Config.forceplate_config['fore_hind_post_cm']/2 for lvl in levels]
    ax.plot(levels, 
             yvals, 
             color=FigConfig.colour_config[clr][2],  
             alpha=0.4, 
             linewidth = 1)

yyyymmdd = '2022-04-0x'  
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_levels_COMBINED')                
# fore-hind and comxy plot means
variable_str = 'CoMy'
title = 'anteroposterior CoS'
meta_p = []

modLINint = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_linear_{variable_str}_{param}_interactionTRUE.csv")
modLINint_aic = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_AICRsq_{variable_str}_{param}_interactionTRUE.csv")
modLIN_aic = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_AICRsq_{variable_str}_{param}.csv")

is_int_aic_better = float(modLINint_aic[modLINint_aic['Model'] == 'Linear'][modLINint_aic['Metric'] == 'AIC']['Value']) < \
                    float(modLIN_aic[modLIN_aic['Model'] == 'Linear'][modLIN_aic['Metric'] == 'AIC']['Value'])
if (modLINint['Pr(>|t|)'][3] < FigConfig.p_thresholds[0]).sum()>0 and is_int_aic_better:
    modLIN = modLINint
    interaction = True
else:
    modLIN = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_linear_{variable_str}_{param}.csv")
    interaction = False

print(f"{variable} is modulated by {(modLIN['Estimate'][1]*Config.forceplate_config['fore_hind_post_cm']*100/2):.2f} ± {(modLIN['Std. Error'][1]*Config.forceplate_config['fore_hind_post_cm']*100/2):.2f} mm/deg")
    
x_centered = df['param'] - np.nanmean(df['param'])
x_pred = np.linspace(np.nanmin(x_centered), np.nanmax(x_centered), endpoint=True)


y_predLIN = (modLIN['Estimate'][0] + modLIN['Estimate'][1]*x_pred + np.nanmean(df[variable]))#*Config.forceplate_config['fore_hind_post_cm']/2
x_pred += np.nanmean(df['param'])
ax.plot(x_pred, 
        y_predLIN, 
        linewidth=2, 
        color=FigConfig.colour_config[clr][2]) 

p_text = title + ' ' + ('*' * (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
if (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
ax.text(0,0.9, p_text, ha = 'center', color = FigConfig.colour_config[clr][2])

legend_colours.append(FigConfig.colour_config[clr][2])
meta_p.append((modLIN['Pr(>|t|)'][2] < FigConfig.p_thresholds).sum())
               
ax.set_ylabel("Centre of support (cm)")
ax.set_xlabel('Incline (deg)')

ax.set_xticks([-40,-20,0,20,40])
ax.set_xlim(-42,40)
ax.set_yticks([-1.5,-1,-0.5,0,0.5,1])
ax.set_title("Slope trials")

tlt = 'Head height'
if np.all(np.diff(meta_p) == 0):
    if meta_p[0] == 0:
        tlt += ' (n.s.)'
    else:
        tlt += f' ({"*" * meta_p[0]})'
else:
    for i, (h, x) in enumerate(zip(meta_p, ['AP', 'ML'])):
        if i == 0:
            tlt += '\n'
        else:
            tlt += ' '
        if h == 0:
            tlt += f'{x}(n.s.)'
        else:
            tlt += f'{x}({"*" * h})'

# p_text_lgdn = ('*' * (modLIN['Pr(>|t|)'][2] < FigConfig.p_thresholds).sum())
# if (modLIN['Pr(>|t|)'][2] < FigConfig.p_thresholds).sum() == 0:
#     p_text_lgdn += "n.s."
    
# lgd = ax.legend([(legend_colours,"solid"), (legend_colours,"dashed")], ['medium', "low"],
#            handler_map={tuple: AnyObjectHandler()}, loc = 'upper center',
#            bbox_to_anchor=(0,-0.5,1,0.2), mode="expand", borderaxespad=0,
#            title = tlt, ncol = 2)
plt.tight_layout(w_pad = 0, pad = 0, h_pad = 0)
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'MS2_{yyyymmdd}_x_CoMy_{param}.svg'))
