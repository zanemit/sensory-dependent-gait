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

fig, ax = plt.subplots(1,1,figsize = (1.2,1.3))

traces_ends = {}

# add forceplate data
df_fp, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="forceplateAngleParams", 
                                               yyyymmdd = '2022-04-04', 
                                               appdx = '_levels')
modLIN = pd.read_csv(Path(Config.paths["forceplate_output_folder"])/"2022-04-04_mixedEffectsModel_linear_snoutBodyAngle_headLVL.csv")
x_centered = np.asarray(df_fp['headLVL'])-np.nanmean(df_fp['headLVL'])
x_pred = np.linspace(x_centered.min(), x_centered.max(), 100)
y_predLIN = modLIN['Estimate'][0] + modLIN['Estimate'][1]*x_pred + np.nanmean(df_fp['snoutBodyAngle'])
x_pred += np.nanmean(df_fp['headLVL'])
ax.plot(x_pred, 
        y_predLIN, 
        linewidth=1, 
        color=FigConfig.colour_config['main'], 
        linestyle = 'dashed',
        label = 'force sensors')
traces_ends[0] = y_predLIN[-1]

yyyymmdd = '2022-08-18'
mod = pd.read_csv(Path(Config.paths["passiveOpto_output_folder"])/f"{yyyymmdd}_mixedEffectsModel_quadratic_preOpto_vs_locom_bodyAngles_incline.csv")

modLINinc = pd.read_csv(Path(Config.paths["passiveOpto_output_folder"])/f"{yyyymmdd}_mixedEffectsModel_linear_snoutBodyAngle_vs_incline_FP_TRDMlocom_TRDMstat_COMPARISON.csv")
modLINinc2 = pd.read_csv(Path(Config.paths["passiveOpto_output_folder"])/f"{yyyymmdd}_mixedEffectsModel_linear_snoutBodyAngle_vs_incline_TRDMlocom_FP_TRDMstat_COMPARISON.csv")

for i, (lnst, data_str, data_col, data_type, lbl) in enumerate((zip(['dotted', 'solid'], 
                                                               ['bodyAngles', 'locomParams'], 
                                                               [0, 'snoutBodyAngle'],
                                                               ['stationary', 'locomoting'],
                                                               ['treadmill stationary', 'treadmill locomoting']
                                                               ))):
    df, yyyymmdd = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                                       dataToLoad = data_str, 
                                                       yyyymmdd = yyyymmdd, 
                                                       appdx = "_incline")
    
    if data_str == 'bodyAngles':
        preOptoAngles = df.loc[:200, (slice(None), slice(None), slice(None), slice(None), 'snoutBody')].mean(axis=0) 
        df = preOptoAngles.reset_index() 
        
    df['headLVL'] = [-int(x[3:]) for x in df['headLVL']] # ensures that "negative" inclines in metadata represent negative slopes

    mice = np.unique(df['mouseID'])

    for m in mice:
        df_sub = df[df['mouseID']==m]
        angles = []
        levels = np.unique(df_sub["headLVL"])
        for lvl in levels:
            angles.append(np.nanmean(df_sub[df_sub["headLVL"] == lvl][data_col]))
        ax.plot(levels, 
                angles, 
                color = FigConfig.colour_config['headbars'], 
                alpha = 0.4, 
                linewidth = 0.5, 
                linestyle = lnst)   

    x_centered = np.asarray(df['headLVL'])-np.nanmean(df['headLVL'])
    x_pred = np.linspace(x_centered.min(), x_centered.max(), endpoint = True)
    if data_type == 'locomoting':
        y_pred = mod['Estimate'][0] + mod['Estimate'][1]*x_pred + mod['Estimate'][2]*x_pred**2 + np.nanmean(df[data_col]) 
    else:
        y_pred = mod['Estimate'][0] + mod['Estimate'][1]*x_pred + mod['Estimate'][2]*x_pred**2 + mod['Estimate'][3] + np.nanmean(df[data_col]) 
    x_pred += np.nanmean(df['headLVL'])
    ax.plot(x_pred, 
            y_pred, 
            linewidth = 1, 
            color = FigConfig.colour_config['headbars'],
            label = lbl,
            linestyle = lnst)
    traces_ends[i+1] = y_pred[-1]
    
p_texts = {} #setupFP_setupTRDMlocom, setupTRDMlocom_setupTRDMstat, setupFP_setupTRDMstat, headHW
p_thresholds = np.asarray(FigConfig.p_thresholds)/3 # pairwise comparison correction
for i, (mod, est) in enumerate(zip(
        [modLINinc, modLINinc2, modLINinc],
        ['setupTRDMlocom', 'setupReleveledTRDMstat', 'setupTRDMstat'])):
    pval = float(mod[mod['Unnamed: 0'] == est]['Pr(>|t|)'])
    if (pval < p_thresholds).sum() != 0:
            p_texts[i] = '*' * (pval < p_thresholds).sum()
    else:
            p_texts[i] = "n.s."
            
stat_dist = [1, 2, 3, 1, 2, 0, 0 ] # x0 : distance from the traces
                                            # x1-x0 : length of first horizontal line
                                            # x2-x1 : length of second horizontal line
                                            # x3 : extension of horizontal lines
                                            # x4 : distance from second horiz line to text
                                            # x5, x6 : y length
p_y_dist = [0.5, 0.5, 0.5]                                    
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
    
    
    

ax.set_xlim(-42,45)
ax.set_ylim(140,180)
ax.set_ylabel('Snout-body angle (deg)')
ax.set_xlabel('Incline (deg)')
# ax.set_title('Unconstrained foot placement')
ax.set_xticks([-40,-20,0,20,40])
ax.set_yticks([140,150,160,170,180])

p_text = ('*' * (mod['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
if (mod['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
ax.text(0,180, f"incline {p_text}", ha = 'center', color = FigConfig.colour_config['headbars'])

p_text_lgdn = ('*' * (mod['Pr(>|t|)'][3] < FigConfig.p_thresholds).sum())
if (mod['Pr(>|t|)'][3] < FigConfig.p_thresholds).sum() == 0:
    p_text_lgdn += "n.s."

# lgd = ax.legend([([FigConfig.colour_config['headbars']],"solid"), 
#                   ([FigConfig.colour_config['headbars']],"dotted")], ['stationary', "locomoting"],
#             handler_map={tuple: AnyObjectHandler()}, loc = 'upper center',
#             bbox_to_anchor=(0.05,-0.5,0.9,0.2), mode="expand", borderaxespad=0,
#             title = f'Data type ({p_text_lgdn})', ncol = 1)

lgd = ax.legend(loc = 'upper center',
                bbox_to_anchor=(-0.2,-0.5,1.2,0.2), #0.055
                mode="expand", 
                borderaxespad=0,
                borderpad = 0.2,
                handlelength = 1.5,
                # title = f"Setup ({p_texts[1]})", 
                ncol = 1)

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_preOpto_vs_locom_bodyAngles_incline.svg", bbox_extra_artists = (lgd, ), bbox_inches = 'tight',dpi=300)

     
