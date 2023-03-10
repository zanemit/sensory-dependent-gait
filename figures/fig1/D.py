import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

param = 'levels'
param_num = 5
legend_colours = []

fig, ax = plt.subplots(1, 1, figsize=(1.4,1.5))
for yyyymmdd, lnst, name in zip(['2022-04-02','2022-04-04'], ['dashed','solid'], ['low', 'medium']):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="meanParamDF", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = f"_{param}")
    mice = np.unique(df['mouse'])
    
    for im, m in enumerate(mice):
        df_sub = df[df['mouse'] == m]
        levels = np.unique(df_sub['param'])
        
        
        for limb_str, limb_clr in zip(['hind_weight_frac', 'fore_weight_frac', 'headplate_weight_frac'],
                                      ['hindlimbs', 'forelimbs', 'headbars']):
            yvals = [np.nanmean(df_sub[df_sub['param']==lvl][limb_str]) for lvl in levels]
            ax.plot(levels, 
                     yvals, 
                     color=FigConfig.colour_config[limb_clr],  
                     alpha=0.4, 
                     linewidth = 0.5,
                     linestyle = lnst)
    # legend_elements.append(Line2D([0], [0], color = FigConfig.colour_config['headplate'], linestyle = lnst, label = name, linewidth = 1))

yyyymmdd = '2022-04-0x'      
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_levels_COMBINED')         
# fore-hind and comxy plot means
meta_p = []
for k in range(2):
    for i, (variable, variable_str, clr) in enumerate(zip(['fore_weight_frac', 'hind_weight_frac', 'headplate_weight_frac'],
                                                          ['foreWfrac', 'hindWfrac', 'headWfrac'],
                                                          ['forelimbs', 'hindlimbs', 'headbars'])):
        modQDR = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_quadratic_{variable_str}_{param}.csv")
        
        x_centered = df['param'] - np.nanmean(df['param'])
        x_pred = np.linspace(np.nanmin(x_centered), np.nanmax(x_centered), endpoint=True)

        if k == 0: # head height 5 is the default condition
            y_predQDR = modQDR['Estimate'][0] + modQDR['Estimate'][1]*x_pred + modQDR['Estimate'][2]*x_pred**2 + np.nanmean(df[variable])
            x_pred += np.nanmean(df['param'])
            ax.plot(x_pred, 
                    y_predQDR, 
                    linewidth=1, 
                    color=FigConfig.colour_config[clr], 
                    linestyle = 'solid') 
        else: # head height 12 is the alternative condition
            y_predQDR = modQDR['Estimate'][0] + modQDR['Estimate'][1]*x_pred + modQDR['Estimate'][2]*x_pred**2 + modQDR['Estimate'][3] + np.nanmean(df[variable])
            x_pred += np.nanmean(df['param'])
            ax.plot(x_pred, 
                    y_predQDR, 
                    linewidth=1, 
                    color=FigConfig.colour_config[clr], 
                    linestyle = 'dashed') 
        p_text = clr + ' ' + ('*' * (modQDR['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
        if (modQDR['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
            p_text += "n.s."
        ax.text(0,2.9-(0.37*i), p_text, ha = 'center', color = FigConfig.colour_config[clr])
        
        if k == 0:
            legend_colours.append(FigConfig.colour_config[clr])
            meta_p.append((modQDR['Pr(>|t|)'][2] < FigConfig.p_thresholds).sum())
               
ax.set_ylabel("Weight fraction")
ax.set_xlabel('Incline (deg)')

ax.set_xticks([-40,-20,0,20,40])
ax.set_xlim(-42,40)
ax.set_yticks([-2,-1,0,1,2,3])
ax.set_title("Incline trials")

tlt = 'Head height'
if np.all(np.diff(meta_p)==0):
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
    
lgd = ax.legend([(legend_colours,"solid"), (legend_colours,"dashed")], ['medium', "low"],
           handler_map={tuple: AnyObjectHandler()}, loc = 'upper center',
           bbox_to_anchor=(0,-0.5,1,0.2), mode="expand", borderaxespad=0,
           title = tlt, ncol = 2)
# plt.tight_layout()

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], yyyymmdd[:-1] + f'x_foreHindHeadWeights_{param}.svg'), bbox_extra_artists = (lgd, ), bbox_inches = 'tight')
