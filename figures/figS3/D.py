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
from figures import scalebars

fig, ax = plt.subplots(2,2,figsize = (3.33/2,1.892), sharex = True, sharey = True)

# TREADMILL
yyyymmdd = '2022-08-18'
param = 'levels'
appdx = '_incline'
conditions = [-40,0,40]
limbs_sub = ['rH1', 'rF1']
colordict = {'rH1': 'homologous', 'rF1': 'diagonal'}      
data = 'preOpto'
irow = 0


for irow, (yyyymmdd,output_folder, appendix, appendix2, param) in enumerate(zip(
        ['2022-08-18','2022-04-0x'],
        ["passiveOpto", "forceplate"],
        ['preOpto_levels_incline', "COMBINED"],
        ['preOpto_levels_incline','forceplate_levels'],
        ['trialType', 'level']
                                       )):
    # PREPARE DATA FOR REGRESSION
    df = pd.read_csv(os.path.join(Config.paths[f"{output_folder}_output_folder"], yyyymmdd + f'_limbPositionRegressionArray_{appendix}.csv'))
    modLIN = pd.read_csv(os.path.join(Config.paths[f"{output_folder}_output_folder"], yyyymmdd + f'_limbPositionRegressionArray_MIXEDMODEL_linear_{appendix2}.csv'))
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
        df[limbCoord + '_cm'] = (df[limbCoord] - model_Yzero)/Config.passiveOpto_config["px_per_cm"][limb]
    
    # BOXPLOTS 
    for ilimb, limb in enumerate(limbs_sub):
        xpos = 0
        limbCoord = limb +'x_cm'
        clr = FigConfig.colour_config[colordict[limb]][2]
        positionsX = np.empty((len(mice), len(conditions)))
        if ilimb != 0:
            ax[irow,ilimb].spines.left.set_visible(False)
            ax[irow,ilimb].get_yaxis().set_visible(False)
        else:
            if ilimb == 0:
                ax[irow,ilimb].text(-0.5,2.4,'anterior', ha = 'right')
                ax[irow,ilimb].text(-0.5,-3.9,'posterior', ha = 'right')
            
        for im, m in enumerate(mice):
            df_sub = df[df['mouseID']== m]
            for ic, c in enumerate(conditions):
                df_sub_cond = df_sub[df_sub[param] == c]
                print(m, limb, df_sub_cond.shape)
                positionsX[im,ic] = np.nanmean(df_sub_cond[limbCoord])
            ax[irow,ilimb].plot([xpos, xpos+1, xpos+2], positionsX[im,:], color = clr, alpha = 0.3)
            ax[irow,ilimb].scatter([xpos, xpos+1, xpos+2], positionsX[im,:], color = clr, alpha = 0.3, s = 2)
        for ic, c in enumerate(conditions):
            positionsX_nonan = positionsX[:,ic][~np.isnan(positionsX[:,ic])]
            ax[irow,ilimb].boxplot(positionsX_nonan, positions = [xpos], medianprops = dict(color = clr, linewidth = 1, alpha = 0.6),
                        boxprops = dict(color = clr, linewidth = 1, alpha = 0.6), capprops = dict(color = clr, linewidth = 1, alpha = 0.6),
                        whiskerprops = dict(color = clr, linewidth = 1, alpha = 0.6), flierprops = dict(mec = clr, linewidth = 1, alpha = 0.6, ms=2))
            ax[irow,ilimb].set_xticks([0,1,2])
            ax[irow,ilimb].set_xticklabels(conditions)
            # ax[ilimb].set_xlabel('Incline (deg)')
            ax[irow,ilimb].set_ylim(-3,2)
            ax[irow,ilimb].set_yticks([-3,-2,-1,0,1,2])
            ax[irow,ilimb].text(1,2, f'{limb[:-1]} {ptext[ilimb]}', ha = 'center', color = clr)
            plt.show()
            xpos = xpos+1 

fig.text(0, 0.5, 'Horizontal position of the foot (cm)', va='center', rotation='vertical')
fig.text(0.56, 0.94, 'force sensors', ha='center')
fig.text(0.56, 0.48, 'passive treadmill', ha='center')
fig.text(0.56, 0, 'Incline (deg)', ha='center')


plt.tight_layout()

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"MS2_{yyyymmdd}_footPositionX_boxplots_{data}_{param}{appdx}.svg", dpi=300)

    
    
