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

yyyymmdd1 = '2022-04-04'
param = 'snoutBodyAngle'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd1, 
                                               appdx = f'_{param}')

mice = np.unique(df['mouse'])
param_num = 5

variable = 'headplate_weight_frac'
clr = 'headbars'
variable_str = 'headWfrac'
minmaxs = ([],[])

fig, ax = plt.subplots(1, 1, figsize=(1.55, 1.5))
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    
    print(np.corrcoef(df_sub['angle'], df_sub[variable])[0,1])
    
    param_split = np.linspace(df_sub['angle'].min()-0.0001, df_sub['angle'].max(), param_num+1)
    xvals = [np.mean((a,b,)) for a,b in zip(param_split[:-1], param_split[1:])]
    df_grouped = df_sub.groupby(pd.cut(df_sub['angle'], param_split)) 
    group_row_ids = df_grouped.groups
    
    minmaxs[0].append(np.nanmin(xvals))
    minmaxs[1].append(np.nanmax(xvals))
    

    yvals = [np.mean(df_sub.loc[val,variable].values) for key,val in group_row_ids.items()]
    ax.plot(xvals, 
             yvals, 
             color=FigConfig.colour_config[clr],  
             alpha=0.4, 
             linewidth = 1)
             
# fore-hind and comxy plot means
modLIN = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd1}_mixedEffectsModel_linear_{variable_str}_{param}.csv")
print(f"{variable} is modulated by {(modLIN['Estimate'][1]*100):.1f} ± {(modLIN['Std. Error'][1]*100):.1f} %/deg")

x_pred = np.linspace(np.nanmin(minmaxs[0])-np.nanmean(df['param']), np.nanmax(minmaxs[1])-np.nanmean(df['param']), endpoint=True)

y_predLIN = modLIN['Estimate'][0] + modLIN['Estimate'][1] * x_pred + np.nanmean(df[variable])
x_pred += np.nanmean(df['param'])
ax.plot(x_pred, y_predLIN, linewidth=2, color=FigConfig.colour_config[clr]) 

# HEAD HEIGHT TRIALS
yyyymmdd2 = '2021-10-26'
df2, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd2, 
                                               appdx = f'_{param}')

modLIN2 = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd2}_mixedEffectsModel_linear_{variable_str}_{param}.csv")
print(f"{variable} is modulated by {(modLIN['Estimate'][1]*100):.1f} ± {(modLIN['Std. Error'][1]*100):.1f} %/deg")

x_pred2 = np.linspace(np.nanmin(df2['angle']), np.nanmax(df2['angle']), endpoint=True)-np.nanmean(df['param'])

y_predLIN2 = modLIN2['Estimate'][0] + modLIN2['Estimate'][1] * x_pred2 + np.nanmean(df2[variable])
x_pred2 += np.nanmean(df2['param'])
ax.plot(x_pred2, y_predLIN2, linewidth=2, color=FigConfig.colour_config['diagonal'][2], linestyle = 'dashed') 

x1 = np.where(x_pred2>x_pred[0])[0][0]
x2 = np.where(x_pred>x_pred2[-1])[0][0]
y_predLIN2_comp = np.linspace(y_predLIN2[x1], y_predLIN2[x2+1], y_predLIN.shape[0])
y_predLIN2_comp-y_predLIN


# TRIAL COMPARISON
modLIN_comb = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd1}_x_{yyyymmdd2}_mixedEffectsModel_linear_headWfrac_vs_snoutBodyAngle_acrossTrialTypes.csv")

for i, txt in enumerate(['angle', 'trial type']):
    p_text = txt + ' ' + ('*' * (modLIN_comb['Pr(>|t|)'][i+1] < FigConfig.p_thresholds).sum())
    if (modLIN_comb['Pr(>|t|)'][i+1] < FigConfig.p_thresholds).sum() == 0:
        p_text += "n.s."
    ax.text(155,0.9-(0.25*i), p_text, ha = 'center', color = FigConfig.colour_config[clr])
            
ax.set_ylabel("Headbar weight fraction")
ax.set_xlabel('Snout-hump angle (deg)')

ax.set_xticks([135,145,155,165,175])
ax.set_xlim(135,175)
ax.set_ylim(-1.5,1)
# ax.set_yticks([-0.5,0,0.5,1.0, 1.5])
ax.set_title("Slope &                     trials", color = FigConfig.colour_config[clr])
ax.text(148, 1.195, "head height", color = FigConfig.colour_config["diagonal"][2])

plt.tight_layout(w_pad = 0, pad = 0, h_pad = 0)
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'MS2_{yyyymmdd2}_foreHindHeadWeights_inclinetrials_{param}.svg'))
