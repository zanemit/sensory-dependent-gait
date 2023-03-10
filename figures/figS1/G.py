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
from figures.fig_config import AnyObjectHandler

param = 'snoutBodyAngle'
param_num = 5
legend_colours = []

fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.5))
minmaxmeans = np.empty((2,3))
minmaxmeans[:,0] = 1000; minmaxmeans[:,1] = 0
for i, (yyyymmdd, lnst) in enumerate(zip(['2022-04-04','2022-04-02'], ['solid','dashed'])):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="meanParamDF", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = f"_{param}")
    mice = np.unique(df['mouse'])
    
    for im, m in enumerate(mice):
        df_sub = df[df['mouse'] == m]
        param_split = np.linspace(df_sub['param'].min()-0.0001, df_sub['param'].max(), param_num+1)
        xvals = [np.mean((a,b,)) for a,b in zip(param_split[:-1], param_split[1:])]
        df_grouped = df_sub.groupby(pd.cut(df_sub['param'], param_split)) 
        group_row_ids = df_grouped.groups
        
        for limb_str, limb_clr in zip(['CoMy_mean', 'CoMx_mean'],
                                      ['main', 'neutral']):
            yvals = [np.mean(df_sub.loc[val,limb_str].values)*Config.forceplate_config['fore_hind_post_cm']/2 for key,val in group_row_ids.items()]
            ax.plot(xvals, 
                    yvals, 
                    color = FigConfig.colour_config[limb_clr],  
                    alpha = 0.4, 
                    linewidth = 0.5, 
                    linestyle = lnst)
        minmaxmeans[i,0] = np.nanmin([minmaxmeans[i,0],np.nanmin(xvals)])
        minmaxmeans[i,1] = np.nanmax([minmaxmeans[i,1],np.nanmax(xvals)])
    minmaxmeans[i,2] = np.nanmean(df['param'])

yyyymmdd = '2022-04-0x'  
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = f'_{param}_COMBINED')                
# fore-hind and comxy plot means
meta_p = []
for k in range(2):
    for i, (variable, variable_str, clr, title) in enumerate(zip(['CoMy_mean', 'CoMx_mean'],
                                                          ['CoMy', 'CoMx'],
                                                          ['main', 'neutral'],
                                                          ['anteroposterior', 'mediolateral'])):
        modLIN = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_linear_{variable_str}_{param}.csv")
        
        x_pred = np.linspace(minmaxmeans[k,0]-minmaxmeans[k,2], minmaxmeans[k,1]-minmaxmeans[k,2], endpoint=True)
        
        if k == 0: # head height 5 is the default condition
            y_predLIN = (modLIN['Estimate'][0] + modLIN['Estimate'][1]*x_pred + np.nanmean(df[variable]))*Config.forceplate_config['fore_hind_post_cm']/2
            x_pred += minmaxmeans[k,2]
            ax.plot(x_pred, 
                    y_predLIN, 
                    linewidth=1, 
                    color=FigConfig.colour_config[clr], 
                    linestyle = 'solid') 
        else: # head height 12 is the alternative condition
            y_predLIN = (modLIN['Estimate'][0] + modLIN['Estimate'][1]*x_pred +  modLIN['Estimate'][2] + np.nanmean(df[variable]))*Config.forceplate_config['fore_hind_post_cm']/2
            x_pred += minmaxmeans[k,2]
            ax.plot(x_pred, 
                    y_predLIN, 
                    linewidth=1, 
                    color=FigConfig.colour_config[clr], 
                    linestyle = 'dashed') 
        
        if variable_str == 'CoMy':
            modLINstats = pd.read_csv(os.path.join(Config.paths["forceplate_output_folder"], yyyymmdd + f'_mixedEffectsModel_linear_{variable_str}_multiPARAM_rH1.csv')) # full model (with incline, angle)
            pos_row = 2
            headheight_row = 4
        else:
            modLINstats = modLIN
            pos_row = 1
            headheight_row = 2
        modLINstats = modLINstats.set_index('Unnamed: 0')
        p_text = title + ' ' + ('*' * (modLINstats['Pr(>|t|)'][pos_row] < FigConfig.p_thresholds).sum())
        if (modLINstats['Pr(>|t|)'][pos_row] < FigConfig.p_thresholds).sum() == 0:
            p_text += "n.s."
        ax.text(160,0.9-(0.18*i), p_text, ha = 'center', color = FigConfig.colour_config[clr])
        
        if k == 0:
            legend_colours.append(FigConfig.colour_config[clr])
            meta_p.append((modLINstats['Pr(>|t|)'][headheight_row] < FigConfig.p_thresholds).sum())
               
ax.set_ylabel("Centre of gravity (cm)")
ax.set_xlabel('Snout-hump angle (deg)')

ax.set_xticks([140,150,160,170,180])
ax.set_xlim(135,180)
ax.set_yticks([-1,-0.5,0,0.5,1])
ax.set_title("Incline trials")

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
    
lgd = ax.legend([(legend_colours,"solid"), (legend_colours,"dashed")], ['medium', "low"],
           handler_map={tuple: AnyObjectHandler()}, loc = 'upper center',
           bbox_to_anchor=(0,-0.6,1,0.3), mode="expand", borderaxespad=0,
           title = tlt, ncol = 2)

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], yyyymmdd[:-1] + f'x_CoMxCoMy_{param}.svg'), bbox_extra_artists = (lgd, ), bbox_inches = 'tight')
