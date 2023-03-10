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

yyyymmdd = '2022-05-06'
fig, ax = plt.subplots(1,1,figsize = (1.7,1.7))
mod = pd.read_csv(Path(Config.paths["mtTreadmill_output_folder"])/f"{yyyymmdd}_mixedEffectsModel_quadratic_snoutBodyAngle_vs_trialType.csv")

data_str = 'strideParams'
data_col = 'snoutBodyAngle'

df, yyyymmdd,refLimb = data_loader.load_processed_data(outputDir = Config.paths["mtTreadmill_output_folder"], 
                                                   dataToLoad = data_str, 
                                                   yyyymmdd = yyyymmdd,
                                                   limb = 'lH1',
                                                   appdx = "")

df['trialType'] = [-int(x[3:]) for x in df['trialType']] # ensures that "negative" inclines in metadata represent negative slopes

mice = np.unique(df['mouseID'])

for m in mice:
    df_sub = df[df['mouseID']==m]
    angles = []
    levels = np.unique(df_sub["trialType"])
    for lvl in levels:
        angles.append(np.nanmean(df_sub[df_sub["trialType"] == lvl][data_col]))
    ax.plot(levels, 
            angles, 
            color = FigConfig.colour_config['headbars'], 
            alpha = 0.4, 
            linewidth = 0.5, 
            linestyle = 'solid')   

x_centered = np.asarray(df['trialType'])-np.nanmean(df['trialType'])
x_pred = np.linspace(x_centered.min(), x_centered.max(), endpoint = True)

y_pred = mod['Estimate'][0] + mod['Estimate'][1]*x_pred + mod['Estimate'][2]*x_pred**2 + np.nanmean(df[data_col]) 
 
x_pred += np.nanmean(df['trialType'])
ax.plot(x_pred, 
        y_pred, 
        linewidth = 1, 
        color = FigConfig.colour_config['headbars'],
        label = 'linear', 
        linestyle = 'solid')


ax.set_xlim(-42,40)
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

p_text_lgdn = ('*' * (mod['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
if (mod['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
    p_text_lgdn += "n.s."
    
plt.tight_layout()

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_snoutBodyAngles_vs_trialType.svg", dpi = 300)

     
