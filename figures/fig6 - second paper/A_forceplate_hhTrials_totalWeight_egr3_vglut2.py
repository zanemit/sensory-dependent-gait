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

param = 'headHW'

fig, ax = plt.subplots(1, 1, figsize=(1.33, 1.4))
ax.hlines(100,xmin=0,xmax=1.2, ls = 'dashed', color = 'grey')

yyyymmdds = ['2023-11-06', '2021-10-26']
ylin_last = []

limb_clr = 'greys'
limb_str = 'headplate_weight_frac'
variable_str = 'headWfrac'

mod_type = 'quadratic'

for yyyymmdd, clr, lbl, lnst in zip(
        yyyymmdds,
        [FigConfig.colour_config['homolateral'][2], FigConfig.colour_config['headbars']],
        ['MS-deficient', 'Control (best fit)'],
        ['solid', 'dashed']
                                    ):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="meanParamDF", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = f'_{param}')
    # CONVERT HEADPLATE VALUES TO PERCENTAGE OF DETECTED WEIGHT
    df[f"{limb_str}_converted"] = -(df[limb_str]-1)*100
    
    mice = np.unique(df['mouse'])
     
    mod = pd.read_csv(os.path.join(Config.paths['forceplate_output_folder'], f"{yyyymmdd}_mixedEffectsModel_{mod_type}_{variable_str}_{param}.csv"), index_col=0)
    meanclr = FigConfig.colour_config['homolateral'][0] if '2023' in yyyymmdd else FigConfig.colour_config[limb_clr][0]
    
    if '2023' in yyyymmdd:
        for im, m in enumerate(mice):
            df_sub = df[df['mouse'] == m]
            headHWs = np.unique(df_sub['param'])
            
            yvals = [np.nanmean(df_sub[df_sub['param']==h][f"{limb_str}_converted"]) for h in np.unique(df_sub['param'])]
            ax.plot(headHWs, 
                     yvals, 
                     color=meanclr,  
                     alpha=0.2, 
                     linewidth = 0.7)
        
        # p_text = '*' * (mod.loc[:,'Pr(>|t|)'].iloc[1] < FigConfig.p_thresholds).sum()
        # ax.text(0.65,190, f"head height: {p_text}", 
        #         ha = 'center', fontsize=5)
        print(mod.loc[:,'Pr(>|t|)'].iloc[1])
        
        ax.text(0.6,188,"MSA-def", fontsize=5,color=meanclr)   
        ax.hlines(186,0.6,1.1,color=meanclr, linestyle='solid', lw=0.7)
    else:
        ax.text(0.06,188,"CTRL", fontsize=5,color=meanclr)
        ax.hlines(186,0.06,0.4,color=meanclr, linestyle='dashed', lw=0.7)

    
    # ADD MEANS
    
    x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)
    x_pred -= np.nanmean(df['param'])
    
    if mod_type == 'linear':
        y_pred = mod.loc['(Intercept)', 'Estimate'] + (mod.loc['param_centred', 'Estimate'] * x_pred) + np.nanmean(df[limb_str])
    else:
        y_pred = mod.loc['(Intercept)', 'Estimate'] + (mod.loc['poly(param_centred, 2, raw = TRUE)1', 'Estimate'] * x_pred) + (mod.loc['poly(param_centred, 2, raw = TRUE)2', 'Estimate'] * x_pred**2) + np.nanmean(df[limb_str])
   
    y_pred = -(y_pred-1)*100
    x_pred += np.nanmean(df['param'])
    ax.plot(x_pred, 
              y_pred, 
              linewidth=1.5, 
              color=meanclr,
              ls=lnst)
    
############# ADD STATS ############# 
statspath = f"C:\\Users\\MurrayLab\\Documents\\Forceplate\\{yyyymmdds[0]}_x_{yyyymmdds[1]}_mixedEffectsModel_{mod_type}_{limb_str}_v_headHW_egr3_vglut2.csv"
stats = pd.read_csv(statspath, index_col =0)
ax.text(0.4, 188, "vs")
ptext = "intercept: "
for j, (ptext, statscol) in enumerate(zip(
        ['intercept: ', 'slopes: '],
        ['trialtypevglut2', 'poly(param_centred, 2, raw = TRUE)1:trialtypevglut2'],
        )):
    p = stats.loc[statscol, "Pr(>|t|)"]
    print(p)
    if (p < np.asarray(FigConfig.p_thresholds)).sum() == 0:
        ptext += "n.s."    
    else:
        ptext += ('*' * (p < np.asarray(FigConfig.p_thresholds)).sum())
    ax.text(0.6,
            165-(j*15),
            ptext, ha='center',
            fontsize=5,
            color=FigConfig.colour_config['homolateral'][0])

############# ADD STATS ############# 

        
ax.set_xlabel('Weight-adjusted\nhead height')

ax.set_xticks([0,0.6,1.2])
ax.set_xlim(-0.1,1.2)
ax.set_yticks([55,100,145,190])
ax.set_ylim(30,190)

ax.set_ylabel("Total leg load (%)")
ax.set_title("Head height trials")
# axes[2].set_ylabel("Total detected weight (%)")
plt.tight_layout()
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplateEGR3_totalWeights_{param}_{limb_str}.svg'),
            transparent = True,
            dpi = 300)
