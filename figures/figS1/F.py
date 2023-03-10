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

fig, axes = plt.subplots(1, 2, figsize=(3, 1.5))
for param, limb_bp,ax in zip(['posXrH1', 'posXrF1'], 
                             ['rH', 'rF'],
                             [axes[0], axes[1]]):
    minmaxmeans = np.empty((2,2))
    minmaxmeans[:,0] = 1000; minmaxmeans[:,1] = 0
    legend_colours = []
    for i, (yyyymmdd, lnst) in enumerate(zip(['2022-04-04','2022-04-02'], ['solid','dashed'])):
        df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                       dataToLoad="meanParamDF", 
                                                       yyyymmdd = yyyymmdd, 
                                                       appdx = f"_{param}")
        mice = np.unique(df['mouse'])
        param_num = 5
        minmaxs = ([],[])
        
        df['param_rescaled'] = (df['param'] - np.nanmean(df['param']))/Config.forceplate_config["px_per_cm"]
    
        for im, m in enumerate(mice):
            df_sub = df[df['mouse'] == m]
            param_split = np.linspace(df_sub['param_rescaled'].min()-0.0001, df_sub['param_rescaled'].max(), param_num+1)
            xvals = [np.mean((a,b,)) for a,b in zip(param_split[:-1], param_split[1:])]
            df_grouped = df_sub.groupby(pd.cut(df_sub['param_rescaled'], param_split)) 
            group_row_ids = df_grouped.groups
            
            minmaxs[0].append(np.nanmin(xvals))
            minmaxs[1].append(np.nanmax(xvals))
            
            for limb_str, limb_clr in zip(['CoMy_mean', 'CoMx_mean'],
                                          ['main', 'neutral']):
                yvals = [np.mean(df_sub.loc[val,limb_str].values)*Config.forceplate_config['fore_hind_post_cm']/2 for key,val in group_row_ids.items()]
                ax.plot(xvals, 
                         yvals, 
                         color=FigConfig.colour_config[limb_clr],  
                         alpha=0.4, 
                         linewidth = 0.5,
                         linestyle = lnst)
            minmaxmeans[i,0] = np.nanmin([minmaxmeans[i,0],np.nanmin(xvals)])
            minmaxmeans[i,1] = np.nanmax([minmaxmeans[i,1],np.nanmax(xvals)])
                 
    yyyymmdd = '2022-04-0x'  
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="meanParamDF", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = f'_{param}_COMBINED') 
    meta_p = []
    for k in range(2):
        for i, (variable, variable_str, clr, title) in enumerate(zip(['CoMy_mean', 'CoMx_mean'],
                                                              ['CoMy', 'CoMx'],
                                                              ['main', 'neutral'],
                                                              ['anteroposterior', 'mediolateral'])):
            modLIN = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_linear_{variable_str}_{param}.csv")
            
            x_pred = np.linspace(minmaxmeans[k,0], minmaxmeans[k,1], endpoint=True) *Config.forceplate_config["px_per_cm"]
            x_centred = df['param'] - np.nanmean(df['param'])
            
            if k == 0: # head height 5 is the default condition
                y_predLIN = (modLIN['Estimate'][0] + modLIN['Estimate'][1]*x_pred + np.nanmean(df[variable]))*Config.forceplate_config['fore_hind_post_cm']/2
                # x_pred += minmaxmeans[k,2]
                ax.plot(x_pred/Config.forceplate_config["px_per_cm"], 
                        y_predLIN, 
                        linewidth=1, 
                        color=FigConfig.colour_config[clr], 
                        linestyle = 'solid') 
            else: # head height 12 is the alternative condition
                y_predLIN = (modLIN['Estimate'][0] + modLIN['Estimate'][1]*x_pred +  modLIN['Estimate'][2] + np.nanmean(df[variable]))*Config.forceplate_config['fore_hind_post_cm']/2
                # x_pred += minmaxmeans[k,2]
                ax.plot(x_pred/Config.forceplate_config["px_per_cm"], 
                        y_predLIN, 
                        linewidth=1, 
                        color=FigConfig.colour_config[clr], 
                        linestyle = 'dashed') 
            
            if variable_str == 'CoMy':    
                modLINstats = pd.read_csv(os.path.join(Config.paths["forceplate_output_folder"], yyyymmdd + f'_mixedEffectsModel_linear_{variable_str}_multiPARAM_{limb_bp}1.csv')) # full model (with incline, angle)
                pos_row = 3
                headheight_row = 4
            else:
                modLINstats = modLIN
                pos_row = 1
                headheight_row = 2
            modLINstats = modLINstats.set_index('Unnamed: 0')
            p_text = title + ' ' + ('*' * (modLINstats['Pr(>|t|)'][pos_row] < FigConfig.p_thresholds).sum())
            if (modLINstats['Pr(>|t|)'][pos_row] < FigConfig.p_thresholds).sum() == 0:
                p_text += "n.s."
            ax.text(-0.5,0.9-(0.18*i), p_text, ha = 'center', color = FigConfig.colour_config[clr])
            
            if k == 0:
                legend_colours.append(FigConfig.colour_config[clr])
                meta_p.append((modLINstats['Pr(>|t|)'][headheight_row] < FigConfig.p_thresholds).sum())
    
    ax.set_xlabel(f'Horizontal {limb_bp} position (cm)')
    
    ax.set_xticks([-3,-2,-1,0,1,2])
    ax.set_xlim(-3,2)
    ax.set_ylim(-1.4,1)
    ax.set_yticks([-1.0,-0.5,0,0.5,1.0])
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
               bbox_to_anchor=(0,-0.5,1.2,0.2), mode="expand", borderaxespad=0,
               title = tlt, ncol = 2)
    
axes[0].set_ylabel("Centre of gravity (cm)")
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], yyyymmdd + f'_CoMxCoMy_posX.svg'), bbox_extra_artists = (lgd, ), bbox_inches = 'tight')
    
