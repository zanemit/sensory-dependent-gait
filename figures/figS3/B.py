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

yyyymmdd = '2022-08-18'
refLimb = 'lH1'
group_num = 6
param = 'headHW'
fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5))
#TODO: should load the combined dataset (shoudl save it from R first!) to have correct centred vals!
df, _ = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                               dataToLoad="locomParams", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_FPcomparison')
modQDRhw = pd.read_csv(Path(Config.paths["passiveOpto_output_folder"])/f"{yyyymmdd}_mixedEffectsModel_quadratic_snoutBodyAngle_vs_headHW_FP_TRDMlocom_TRDMstat_COMPARISON.csv")
modQDRhw2 = pd.read_csv(Path(Config.paths["passiveOpto_output_folder"])/f"{yyyymmdd}_mixedEffectsModel_quadratic_snoutBodyAngle_vs_headHW_TRDMlocom_FP_TRDMstat_COMPARISON.csv")

minmaxs = np.asarray((1000, 0), dtype = np.float32)

setup = 'TRDMstat'

# arr = np.empty((len(Config.passiveOpto_config['mice']), group_num))

df_subx = df[df['setup'] == setup]
for im, m in enumerate(Config.passiveOpto_config['mice']):
    df_sub = df_subx[df_subx['mouseID'] == m]
    param_split = np.linspace(df_sub[param].min()-0.0001, df_sub[param].max(), group_num+1)
    xvals = [np.nanmean((a,b,)) for a,b in zip(param_split[:-1], param_split[1:])]
    df_grouped = df_sub.groupby(pd.cut(df_sub[param], param_split)) 
    group_row_ids = df_grouped.groups
    
    yvals = [np.nanmean(df_sub.loc[val,'snoutBodyAngle'].values) for key,val in group_row_ids.items()]
    # arr[im, :] = yvals
    ax.plot(xvals, 
            yvals, 
            color = FigConfig.colour_config['headbars'],  
            alpha = 0.4, 
            linewidth = 1)
    minmaxs[0] = np.nanmin([minmaxs[0],np.nanmin(xvals)])
    minmaxs[1] = np.nanmax([minmaxs[1],np.nanmax(xvals)])

# plt.plot(xvals, np.nanmean(arr, axis = 0), color = 'red')
# model: snoutBodyAngle ~ poly(headHW, 2) + setup + 1|mouseID
x_pred = np.linspace(minmaxs[0]- np.nanmean(df[param]), minmaxs[1]- np.nanmean(df[param]), 100)

y_predQDR_FP = modQDRhw['Estimate'][0] + modQDRhw['Estimate'][1]*x_pred + modQDRhw['Estimate'][2]*x_pred**2 + np.nanmean(df['snoutBodyAngle'])
y_predQDR_TRDM_preOpto = modQDRhw['Estimate'][0] + modQDRhw['Estimate'][1]*x_pred + modQDRhw['Estimate'][2]*(x_pred**2) + modQDRhw['Estimate'][4] + np.nanmean(df['snoutBodyAngle']) 

x_pred += np.nanmean(df['headHW'])
# p_texts = np.empty(2, dtype = 'object')
# for i, est in enumerate([1,3]):
#     if (modQDRhw['Pr(>|t|)'][est] < FigConfig.p_thresholds).sum() != 0:
#         p_texts[i] = '*' * (modQDRhw['Pr(>|t|)'][est] < FigConfig.p_thresholds).sum()
#     else:
#         p_texts[i] = "n.s."

traces_ends = {}
for i, (y_pred, lnst, lc, lbl) in enumerate(zip(
        [y_predQDR_FP, y_predQDR_TRDM_preOpto], 
        ['dashed', 'solid'],
        [FigConfig.colour_config['diagonal'][2], FigConfig.colour_config['headbars']],
        ['force sensors', 'treadmill stationary'])):
    ax.plot(x_pred, 
            y_pred, 
            color = lc, 
            linewidth = 2, 
            linestyle = lnst,
            label = lbl)
    traces_ends[i] = y_pred[-1]
    
p_texts = {} #setupFP_setupTRDMlocom, setupTRDMlocom_setupTRDMstat, setupFP_setupTRDMstat, headHW
p_thresholds = np.asarray(FigConfig.p_thresholds)/3 # pairwise comparison correction
for i, (mod, est) in enumerate(zip(
        [modQDRhw, modQDRhw2, modQDRhw, modQDRhw],
        ['setupTRDMlocom', 'setupReleveledTRDMstat', 'setupTRDMstat', 'poly(headHW_centred, 2, raw = TRUE)1'])):
    pval = float(mod[mod['Unnamed: 0'] == est]['Pr(>|t|)'])
    if (pval < p_thresholds).sum() != 0:
            p_texts[i] = '*' * (pval < p_thresholds).sum()
    else:
            p_texts[i] = "n.s."
            
stat_dist = [0.01, 0.03, 0.05, 0.08, 0.02, 4, -4 ] # x0 : distance from the traces
                                            # x1-x0 : length of first horizontal line
                                            # x2-x1 : length of second horizontal line
                                            # x3 : extension of horizontal lines
                                            # x4 : distance from second horiz line to text
                                            # x5, x6 : y length
p_y_dist = [0.5, 2, 0.5]                                    
for iy, y_tr in enumerate(traces_ends.values()):
    if iy < len(traces_ends)-1:
        ax.hlines(y_tr, xmin = x_pred[-1]+stat_dist[0], xmax = x_pred[-1]+stat_dist[1], linewidth = 0.5, color = 'black')
        ax.vlines(x_pred[-1]+stat_dist[1], ymin = traces_ends[iy], ymax = traces_ends[iy+1], linewidth = 0.5, color = 'black')
        ax.hlines(np.mean((traces_ends[iy], traces_ends[iy+1])), xmin = x_pred[-1]+stat_dist[1], xmax = x_pred[-1]+stat_dist[2], linewidth = 0.5, color = 'black')
        ax.vlines(x_pred[-1]+stat_dist[2], ymin = np.mean((traces_ends[iy], traces_ends[iy+1])), ymax = np.mean((traces_ends[iy], traces_ends[iy+1]))+stat_dist[iy+5], linewidth = 0.5, color = 'black')
        ax.hlines(np.mean((traces_ends[iy], traces_ends[iy+1]))+stat_dist[iy+5], xmin = x_pred[-1]+stat_dist[2], xmax = x_pred[-1]+stat_dist[2]+stat_dist[1]-stat_dist[0], linewidth = 0.5, color = 'black')
        ax.text(x_pred[-1]+stat_dist[2]+stat_dist[1], np.mean((traces_ends[iy], traces_ends[iy+1]))+stat_dist[iy+5]-p_y_dist[iy], p_texts[iy])
    else:
        ax.hlines(y_tr, xmin = x_pred[-1]+stat_dist[0], xmax = x_pred[-1]+stat_dist[1], linewidth = 0.5, color = 'black', linestyle = 'solid')
        ax.hlines(y_tr, xmin = x_pred[-1]+stat_dist[3], xmax = x_pred[-1]+stat_dist[1]+stat_dist[3], linewidth = 0.5, color = 'black', linestyle = 'dashed')
        ax.hlines(traces_ends[0], xmin = x_pred[-1]+stat_dist[3], xmax = x_pred[-1]+stat_dist[1]+stat_dist[3], linewidth = 0.5, color = 'black', linestyle = 'dashed')
        ax.vlines(x_pred[-1]+stat_dist[1]+stat_dist[3], ymin = traces_ends[iy], ymax = traces_ends[0], linewidth = 0.5, color = 'black', linestyle = 'dashed')
        ax.hlines(np.mean((traces_ends[iy], traces_ends[0])), xmin = x_pred[-1]+stat_dist[1]+stat_dist[3], xmax = x_pred[-1]+stat_dist[2]+stat_dist[3], linewidth = 0.5, color = 'black', linestyle = 'dashed')
        ax.text(x_pred[-1]+stat_dist[2]+stat_dist[3]+stat_dist[4], np.mean((traces_ends[iy], traces_ends[0]))-0.3, p_texts[iy])

ax.text(0.6, 180, f"head height {p_texts[3]}", ha = 'center', color = FigConfig.colour_config['headbars'])

ax.set_xlim(0,1.2)
ax.set_ylim(138,180)
ax.set_ylabel('Snout-body angle (deg)')
ax.set_xlabel('Weight-adjusted head height')
ax.set_xticks([0,0.4,0.8,1.2])
ax.set_yticks([140,150,160,170,180])

lgd = ax.legend(loc = 'upper center',
                bbox_to_anchor=(0.1,-0.5,0.9,0.2), #0.055
                mode="expand", 
                borderaxespad=0,
                borderpad = 0.2,
                handlelength = 1.5,
                # title = f"Setup ({p_texts[1]})", 
                ncol = 1)

lgd._legend_box.align = 'center'

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"MS2_{yyyymmdd}_snoutBodyAngles_vs_headHW_treadmill.svg",
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight')



