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


fig, ax = plt.subplots(2, 1, figsize=(1.5, 3))

# ------ Head height trials ------
yyyymmdd = '2021-10-26'  
param = '_snoutBodyAngle'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = param)  
 
ax[0].axvline(140,ymax = 0.92, linestyle = 'dashed', color = FigConfig.colour_config['neutral'])
ax[0].axvline(170, ymax = 0.92,linestyle = 'dashed', color = FigConfig.colour_config['neutral'])

for i, (variable, variable_str, clr) in enumerate(zip(['headplate_weight_frac','fore_weight_frac', 'hind_weight_frac'],
                                                      ['headWfrac', 'foreWfrac', 'hindWfrac'],
                                                      ['headplate', 'forelimbs', 'hindlimbs'])):
    modLIN = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_linear_{variable_str}_{param}.csv")
    
    x_centered = df['param'] - np.nanmean(df['param'])
    x_pred = np.linspace(np.nanmin(x_centered), np.nanmax(x_centered), endpoint=True)
    
    if i == 0:
        headplate_zero_angle = (-(np.nanmean(df[variable]) + modLIN['Estimate'][0]) / modLIN['Estimate'][1]) + np.nanmean(df['param'])
        ax[0].axvline(headplate_zero_angle, ymax = 0.92, linestyle = 'dashed', color = FigConfig.colour_config['neutral'])
        ax[0].axhline(0, 0, (headplate_zero_angle-x_pred[0]-np.nanmean(df['param']))/(x_pred[-1]-x_pred[0]), linestyle = 'dashed', color = FigConfig.colour_config['neutral'])
    
    y_predLIN = modLIN['Estimate'][0] + modLIN['Estimate'][1] * x_pred + np.nanmean(df[variable])
    x_pred += np.nanmean(df['param'])
    ax[0].plot(x_pred, y_predLIN, linewidth=2, color=FigConfig.colour_config[clr])  

ax[0].set_ylabel("Weight fraction")
ax[0].set_xlabel('Snout-hump angle (deg)')

ax[0].set_xticks([135,145,155,165,175])
ax[0].set_xlim(135,175)
ax[0].set_yticks([-0.5,0,0.5,1.0, 1.5])
ax[0].set_title("Head height trials")

ax[0].text(140,1.38,'a', ha = 'center', color = FigConfig.colour_config['neutral'])
ax[0].text(headplate_zero_angle,1.38,'b', ha = 'center', color = FigConfig.colour_config['neutral'])
ax[0].text(170,1.38,'c', ha = 'center', color = FigConfig.colour_config['neutral'])

# ------ Incline trials ------
yyyymmdd = '2022-04-04'  
param = '_levels'    
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = param)     
    
ax[1].axvline(-36,ymax = 0.92, linestyle = 'dashed', color = FigConfig.colour_config['neutral'])
ax[1].axvline(36,ymax = 0.92, linestyle = 'dashed', color = FigConfig.colour_config['neutral'])

for i, (variable, variable_str, clr) in enumerate(zip(['fore_weight_frac', 'hind_weight_frac', 'headplate_weight_frac'],
                                                      ['foreWfrac', 'hindWfrac', 'headWfrac'],
                                                      ['forelimbs', 'hindlimbs', 'headplate'])):
    modQDR = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_quadratic_{variable_str}_{param}.csv")
    
    x_centered = df['param'] - np.nanmean(df['param'])
    x_pred = np.linspace(np.nanmin(x_centered), np.nanmax(x_centered), endpoint=True)

    y_predQDR = modQDR['Estimate'][0] + modQDR['Estimate'][1]*x_pred + modQDR['Estimate'][2]*x_pred**2 + np.nanmean(df[variable])
    x_pred += np.nanmean(df['param'])
    ax[1].plot(x_pred, 
               y_predQDR, 
               linewidth=2, 
               color=FigConfig.colour_config[clr], 
               linestyle = 'solid') 
              
ax[1].set_ylabel("Weight fraction")
ax[1].set_xlabel('Incline (deg)')

ax[1].set_xticks([-40,-20,0,20,40])
ax[1].set_xlim(-42,40)
ax[1].set_yticks([-1,-0.5,0,0.5,1,1.5])
ax[1].set_title("Incline trials")

ax[1].text(-36,1.38,'d', ha = 'center', color = FigConfig.colour_config['neutral'])
ax[1].text(36,1.38,'e', ha = 'center', color = FigConfig.colour_config['neutral'])

fig.tight_layout(h_pad = 15)

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], 'forceplate_explanation_schematic.svg'))
