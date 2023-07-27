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

yyyymmdd = '2021-10-26'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="forceplateAngleParams", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_headHW')
modQDRhw = pd.read_csv(Path(Config.paths["forceplate_output_folder"])/f"{yyyymmdd}_mixedEffectsModel_quadratic_snoutBodyAngle_headHW.csv")

mice = np.unique(df['mouseID'])

fig, ax = plt.subplots(1, 1, figsize=(1.55, 1.5))
for m in mice:
    df_sub = df[df['mouseID'] == m]
    headHWs = np.unique(df_sub['headHW'])
    bodyAngles = [np.nanmean(df_sub[df_sub['headHW']==h]['snoutBodyAngle']) for h in np.unique(df_sub['headHW'])]
    ax.plot(headHWs, 
            bodyAngles, 
            color = FigConfig.colour_config['headbars'], 
            alpha=0.4, 
            linewidth = 1)

x_centered = np.asarray(df['headHW'])-np.nanmean(df['headHW'])
x_pred = np.linspace(x_centered.min(), x_centered.max(), 100)
y_predQDR = modQDRhw['Estimate'][0] + modQDRhw['Estimate'][1]*x_pred + modQDRhw['Estimate'][2]*x_pred**2 + np.nanmean(df['snoutBodyAngle'])
x_pred += np.nanmean(df['headHW'])
p_text = '*' * (modQDRhw['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum()
ax.plot(x_pred, 
         y_predQDR, 
         color = FigConfig.colour_config['headbars'], 
         linewidth = 2, 
         label = p_text)

ax.set_xlim(0,1.2)
ax.set_ylim(140,180)
ax.set_ylabel('Snout-hump angle (deg)')
ax.set_xlabel('Weight-adjusted head height')
ax.set_title('Head height trials')
ax.set_xticks([0,0.4,0.8,1.2])
ax.set_yticks([140,150,160,170,180])
ax.text(0.6,179, f"head height {p_text}", ha = 'center', color = FigConfig.colour_config['headbars'])

plt.tight_layout(w_pad = 0, pad = 0, h_pad = 0)
fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"MS2_{yyyymmdd}_snoutBodyAngles_vs_headHW.svg")



