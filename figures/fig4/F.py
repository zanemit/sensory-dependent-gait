import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from copy import deepcopy
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

from matplotlib import rcParams
rcParams['axes.spines.right'] = True

param = 'snoutBodyAngle'

xmin = 140
xmax = 175
interval = 5
xlbl = 'Snout-hump angle (deg)'
p_lbl = 'body tilt'

legend_colours = []
ps = []

x = np.arange(xmin, xmax+interval, interval)

fig, ax = plt.subplots(1,1, figsize = (1.3,1.4))
ax2 = ax.twinx()
for i, (yyyymmdd, outputDir, lnst, appdx) in enumerate(zip(
        ['2022-08-18', '2021-10-23'],
        [Config.paths["passiveOpto_output_folder"],Config.paths["mtTreadmill_output_folder"]],
        ['solid', 'dashed'],
        ['', '']
        )):
    
    filepath = os.path.join(outputDir, f"{yyyymmdd}_dutyCyclesReduced{appdx}.csv")

    df_reduced = pd.read_csv(filepath)

    mice = np.unique(df_reduced['mouseID'])
    # limbs = np.unique(df['limb']) 
    pairs = ['Fore','Hind']
    
    # df_reduced = deepcopy(df[df['limb'] == 'lH'])
    # df_reduced.reset_index(inplace=True) 
    # df_reduced.drop(['dutyCycle', 'limb', 'index', 'Unnamed: 0'], axis = 1, inplace=True)
    
    # for pair in pairs:
    #     df_reduced[f'{pair}_dutyCycle'] = np.nanmean(np.vstack((
    #                             df[df['limb'] == f'l{pair[0]}']['dutyCycle'],
    #                             df[df['limb'] == f'r{pair[0]}']['dutyCycle'])).T, 
    #                             axis = 1)
    # # df_reduced['Diff_dutyCycle'] = df_reduced['Hind_dutyCycle']-df_reduced['Fore_dutyCycle']
    # df_reduced['Diff_dutyCycle'] = np.asarray(df_reduced['Hind_dutyCycle']/df_reduced['Fore_dutyCycle'])
    # df_reduced['Diff_dutyCycle'].replace(0, np.nan, inplace=True)
    # df_reduced.to_csv(os.path.join(outputDir, f"{yyyymmdd}_dutyCyclesReduced{appdx}.csv"))
    
    # COMPUTE MEAN DUTY CYCLES PER MOUSE PER PARAM INTERVAL (for plotting)   
    arr_3d = np.empty((len(pairs), len(x), len(mice)))
    
    for im, m in enumerate(mice):
        df_sub = df_reduced[df_reduced['mouseID']==m]
        for ixid, ix in enumerate(x):
            df_sub_x = df_sub[(df_sub[param]>=(ix-(interval/2)))&(df_sub[param]<(ix+(interval/2)))]
            for ip, pair in enumerate(pairs):
                arr_3d[ip, ixid, im] = np.nanmean(df_sub_x[f'{pair}_dutyCycle'])
    
    hind_id = np.where(['H' in x for x in pairs])[0][0]
    fore_id = np.where(['H' not in x for x in pairs])[0][0]
    
    for ids, clr in zip([hind_id, fore_id],
                        ['hindlimbs', 'forelimbs']):
        duty_mean = np.nanmean(arr_3d[ids,:,:], axis = 1) # limbs x intervals
        duty_sem =  scipy.stats.sem(arr_3d[ids,:,:], axis = 1, nan_policy = 'omit')
        duty_ci = duty_sem * scipy.stats.t.ppf((1 + 0.95) / 2., arr_3d[ids,:,:].shape[1]-1)
    
        ax.fill_between(x, 
                        duty_mean - duty_ci, 
                        duty_mean + duty_ci, 
                        facecolor = FigConfig.colour_config[clr], 
                        alpha = 0.2)
        ax.plot(x, 
                duty_mean, 
                color = FigConfig.colour_config[clr],
                linestyle = lnst)
        
        if i == 0:
            legend_colours.append(FigConfig.colour_config[clr])
    
    ax.set_ylim(0,1)
    ax.set_xlim(xmin,xmax)
    ax.set_ylabel("Forelimb duty cycle", color = FigConfig.colour_config[clr])
    ax.set_xlabel(xlbl)
    ax.text(126, 0.15, "Hindlimb duty cycle",color = FigConfig.colour_config['hindlimbs'], rotation = 90)
    
    arr_3d_pair = arr_3d[hind_id,:,:] / arr_3d[fore_id,:,:]
    duty_mean = np.nanmean(arr_3d_pair, axis = 1) # limbs x intervals
    duty_sem =  scipy.stats.sem(arr_3d_pair, axis = 1, nan_policy = 'omit')
    duty_ci = duty_sem * scipy.stats.t.ppf((1 + 0.95) / 2., arr_3d_pair.shape[1]-1)
    
    ax2.fill_between(x, 
                     duty_mean - duty_ci, 
                     duty_mean + duty_ci, 
                     facecolor = FigConfig.colour_config['greys'][2], 
                     alpha = 0.2)
    ax2.plot(x, 
             duty_mean, 
             color = FigConfig.colour_config['greys'][2],
             linestyle = lnst)
    
    if i == 0:
        legend_colours.append(FigConfig.colour_config['greys'][2])
    
    ax2.set_ylim(0.90,1.40)
    ax2.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    ax2.set_ylabel("Hind:fore duty cycle ratio", color = FigConfig.colour_config['greys'][2])
    
    statspath = os.path.join(outputDir, f"{yyyymmdd}_mixedEffectsModel_dutyCycles_linear_{param}_speed_acceleration{appdx}.csv")
    stats = pd.read_csv(statspath)
    
    param_row = 1 # snoutBodyAngle is in row1 of stats (file name lists them in order)
    
    p_text = p_lbl + ' ' + ('*' * (stats.loc[param_row,'Pr(>|t|)'] < FigConfig.p_thresholds).sum())
    if (stats.loc[param_row,'Pr(>|t|)'] < FigConfig.p_thresholds).sum() == 0:
        p_text += "n.s."
    ps.append(p_text)
    print(f"{i}: regression incline {stats.loc[param_row,'Estimate']} ± {stats.loc[param_row,'Std. Error']}, p-value {stats.loc[param_row,'Pr(>|t|)']}")
    if i == 1:
        if p_text == ps[0]:
            continue
        else:
            ax.hlines(1.03+0.025, 
                      xmin = 0.10, 
                      xmax = 0.23, 
                      linestyle = 'solid', 
                      color = FigConfig.colour_config['greys'][2], 
                      linewidth = 1, 
                      transform=ax.transAxes,
                      clip_on = False)
            ax.hlines(1.03+0.025-(i*0.11), 
                      xmin = 0.10, 
                      xmax = 0.23, 
                      linestyle = 'dashed', 
                      color = FigConfig.colour_config['greys'][2], 
                      linewidth = 1,
                      transform=ax.transAxes)
    ax.text(0.27, 1.03-(i*0.11), p_text, ha = 'left', 
            color = FigConfig.colour_config['greys'][2], 
            transform=ax.transAxes)

lgd = ax.legend([(legend_colours,"solid"), (legend_colours,"dashed")], 
                ['passive treadmill', "motorised treadmill"],ncol = 1,
                handler_map={tuple: AnyObjectHandler()}, loc = 'upper center',
                bbox_to_anchor=(-0.1,-0.5,1.1,0.2), mode="expand", borderaxespad=0)

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_dutyCycles_vs_{param}{appdx}.svg", dpi=300, bbox_extra_artists = (lgd, ), bbox_inches = 'tight')

