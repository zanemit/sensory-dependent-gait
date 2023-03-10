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

fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.5))
for yyyymmdd, lnst, name in zip(['2022-04-02','2022-04-04'], ['dashed','solid'], ['low', 'medium']):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="forceplateAngleParams", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = '_levels')
    
    mice = np.unique(df['mouseID'])
    
    for m in mice:
        df_sub = df[df['mouseID'] == m]
        levels = np.unique(df_sub['headLVL'])
        bodyAngles = [np.nanmean(df_sub[df_sub['headLVL']==h]['snoutBodyAngle']) for h in np.unique(df_sub['headLVL'])]
        ax.plot(levels, 
                bodyAngles, 
                color = FigConfig.colour_config['headbars'], 
                alpha=0.4, 
                linewidth = 0.5, 
                linestyle = lnst)

yyyymmdd = '2022-04-0x'                 
# fore-hind and comxy plot means
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="forceplateAngleParams", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_levels_COMBINED')
modLINint = pd.read_csv(Path(Config.paths["forceplate_output_folder"])/f"{yyyymmdd}_mixedEffectsModel_linear_snoutBodyAngle_headLVL_interactionTRUE.csv")
modLINint_aic = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_AICRsq_snoutBodyAngle_headLVL_interactionTRUE.csv")
modLIN_aic = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_AICRsq_snoutBodyAngle_headLVL.csv")

is_int_aic_better = float(modLINint_aic[modLINint_aic['Model'] == 'Linear'][modLINint_aic['Metric'] == 'AIC']['Value']) < \
                     float(modLIN_aic[modLIN_aic['Model'] == 'Linear'][modLIN_aic['Metric'] == 'AIC']['Value'])

if (modLINint['Pr(>|t|)'][3] < FigConfig.p_thresholds[0]).sum()>0 and is_int_aic_better:
    modLIN = modLINint
    interaction = True
else:
    modLIN = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_linear_snoutBodyAngle_headLVL.csv")
    interaction = False

x_centered = np.asarray(df['headLVL'])-np.nanmean(df['headLVL'])
x_pred = np.linspace(x_centered.min(), x_centered.max(), 100)
for k in range(2):
    if k == 0: # head height 5 is the default condition
        y_predLIN = modLIN['Estimate'][0] + modLIN['Estimate'][1]*x_pred + np.nanmean(df['snoutBodyAngle'])
        x_pred += np.nanmean(df['headLVL'])
        ax.plot(x_pred, 
                y_predLIN, 
                linewidth=1, 
                color=FigConfig.colour_config['headbars'], 
                linestyle = 'solid') 
    else: # head height 12 is the alternative condition
        if interaction:
            y_predLIN = modLIN['Estimate'][0] + modLIN['Estimate'][1]*x_pred + modLIN['Estimate'][2] + modLIN['Estimate'][3]*x_pred + np.nanmean(df['snoutBodyAngle'])
           
        else:
            y_predLIN = modLIN['Estimate'][0] + modLIN['Estimate'][1]*x_pred + modLIN['Estimate'][2] + np.nanmean(df['snoutBodyAngle'])
        x_pred += np.nanmean(df['headLVL'])
        ax.plot(x_pred, 
                y_predLIN, 
                linewidth=1, 
                color=FigConfig.colour_config['headbars'], 
                linestyle = 'dashed') 

ax.set_xlim(-42,40)
ax.set_ylim(140,180)
ax.set_ylabel('Snout-hump angle (deg)')
ax.set_xlabel('Incline (deg)')
ax.set_title('Incline trials')
ax.set_xticks([-40,-20,0,20,40])
ax.set_yticks([140,150,160,170,180])
p_text = ('*' * (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
if (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
ax.text(0,177, f"incline {p_text}", ha = 'center', color = FigConfig.colour_config['headbars'])

p_text_lgdn = ('*' * (modLIN['Pr(>|t|)'][2] < FigConfig.p_thresholds).sum())
if (modLIN['Pr(>|t|)'][2] < FigConfig.p_thresholds).sum() == 0:
    p_text_lgdn += "n.s."

lgd = ax.legend([([FigConfig.colour_config['headbars']],"solid"), 
                 ([FigConfig.colour_config['headbars']],"dashed")], ['medium', "low"],
           handler_map={tuple: AnyObjectHandler()}, loc = 'upper center',
           bbox_to_anchor=(0,-0.5,1,0.2), mode="expand", borderaxespad=0,
           title = f'Head height ({p_text_lgdn})', ncol = 2)
fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_snoutBodyAngles_vs_headLVL.svg", bbox_extra_artists = (lgd, ), bbox_inches = 'tight')



