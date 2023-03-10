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
from figures import scalebars

# TODO generate both preOpto and locom
# add limb labels
# make a similar plot for the forceplate data (rF1 and rH1)
# make a forceplate data plot with GRF centre vs foot position averaged across inclines (head high and low + rF1 and rH1)

yyyymmdd = '2022-05-06'
param = 'trialType'
conditions = [-40,0,40]
limbs_sub = ['rH1', 'rF1']
colordict = { 'rH1': 'homologous',  'rF1': 'diagonal'}      


# PREPARE DATA FOR REGRESSION
df = pd.read_csv(os.path.join(Config.paths["mtTreadmill_output_folder"], yyyymmdd + '_limbPositionRegressionArray_egocentric.csv'))
modLIN = pd.read_csv(os.path.join(Config.paths["mtTreadmill_output_folder"], yyyymmdd + f'_limbPositionRegressionArray_MIXEDMODEL_linear_mtTreadmill_levels.csv'))
modLIN = modLIN.set_index('Unnamed: 0')

df[param] = [-int(x[3:]) for x in df[param]]

mice = np.unique(df['mouseID'])
levels_centred = df[param] - np.nanmean(df[param])
xvals = np.linspace(np.nanmin(levels_centred), np.nanmax(levels_centred))

ptext = []
for ilimb, limb in enumerate(limbs_sub):
    limbCoord = limb + 'x'

    # convert y axis to zero incline-centred cm 
    zero_incline_centred = - np.nanmean(df[param])
    model_Yzero = modLIN.loc[f'Intercept_{limbCoord}','Estimate'] + modLIN.loc[f'param_centred_{limbCoord}','Estimate'] * zero_incline_centred + np.nanmean(df[limbCoord])

    if (modLIN.loc[f'param_centred_{limbCoord}','Pr(>|t|)'] < FigConfig.p_thresholds).sum() == 0:
        ptext.append( "n.s." )    
    else:
        ptext.append('*' * (modLIN.loc[f'param_centred_{limbCoord}','Pr(>|t|)'] < FigConfig.p_thresholds).sum())
    # add rescaled positions to the df
    df[limbCoord + '_cm'] = (df[limbCoord] - model_Yzero)/Config.mtTreadmill_config["px_per_cm"][limb]

# BOXPLOTS 
fig, ax = plt.subplots(1,2,figsize = (1.8,1.8))
for ilimb, limb in enumerate(limbs_sub):
    xpos = 0
    limbCoord = limb +'x_cm'
    clr = FigConfig.colour_config[colordict[limb]][2]
    positionsX = np.empty((len(mice), len(conditions)))
    if ilimb != 0:
        ax[ilimb].spines.left.set_visible(False)
        ax[ilimb].get_yaxis().set_visible(False)
    else:
        ax[ilimb].set_ylabel('Horizontal position \n of the foot (cm)')
        ax[ilimb].text(-0.5,2.3,'anterior', ha = 'right')
        ax[ilimb].text(-0.5,-3.6,'posterior', ha = 'right')
        
    for im, m in enumerate(mice):
        df_sub = df[df['mouseID']== m]
        for ic, c in enumerate(conditions):
            df_sub_cond = df_sub[df_sub[param] == c]
            positionsX[im,ic] = np.nanmean(df_sub_cond[limbCoord])
        ax[ilimb].plot([xpos, xpos+1, xpos+2], positionsX[im,:], color = clr, alpha = 0.3)
        ax[ilimb].scatter([xpos, xpos+1, xpos+2], positionsX[im,:], color = clr, alpha = 0.3, s = 2)
    for ic, c in enumerate(conditions):
        ax[ilimb].boxplot(positionsX[:,ic], positions = [xpos], medianprops = dict(color = clr, linewidth = 1, alpha = 0.6),
                    boxprops = dict(color = clr, linewidth = 1, alpha = 0.6), capprops = dict(color = clr, linewidth = 1, alpha = 0.6),
                    whiskerprops = dict(color = clr, linewidth = 1, alpha = 0.6), flierprops = dict(mec = clr, linewidth = 1, alpha = 0.6, ms=2))
        ax[ilimb].set_xticks([0,1,2])
        ax[ilimb].set_xticklabels(conditions)
        ax[ilimb].set_xlabel('Incline (deg)')
        ax[ilimb].set_ylim(-3,2)
        ax[ilimb].set_yticks([-3,-2,-1,0,1,2])
        ax[ilimb].text(1,2, f'{limb[:-1]} {ptext[ilimb]}', ha = 'center', color = clr)
        
        xpos = xpos+1 

plt.tight_layout()

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_footPositionX_boxplots_mtTreadmill_{param}.svg", dpi=300)
