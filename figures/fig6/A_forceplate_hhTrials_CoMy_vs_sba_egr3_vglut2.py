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

param = 'snoutBodyAngle'

fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.4))

yyyymmdds = ['2023-11-06', '2021-10-26']
ylin_last = []

limb_clr = 'greys'
limb_str = 'CoMy_mean'
variable_str = 'CoMy'

mod_type = 'linear'

for yyyymmdd,  lnst in zip(
        yyyymmdds,
        ['solid', 'dashed']
                                    ):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="meanParamDF", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = f'_{param}')
    
    mice = np.unique(df['mouse'])
     
    mod = pd.read_csv(os.path.join(Config.paths['forceplate_output_folder'], f"{yyyymmdd}_mixedEffectsModel_{mod_type}_{variable_str}_{param}.csv"), index_col=0)
    meanclr = FigConfig.colour_config['homolateral'][0] if '2023' in yyyymmdd else FigConfig.colour_config[limb_clr][0]
    
    param_num = 3
    minmaxs = ([],[])
    for im, m in enumerate(mice):
        df_sub = df[df['mouse'] == m]
        param_split = np.linspace(df_sub['param'].min()-0.0001, df_sub['param'].max(), param_num+1)
        xvals = [np.mean((a,b,)) for a,b in zip(param_split[:-1], param_split[1:])]
        df_grouped = df_sub.groupby(pd.cut(df_sub['param'], param_split)) 
        group_row_ids = df_grouped.groups
        
        minmaxs[0].append(np.nanmin(xvals))
        minmaxs[1].append(np.nanmax(xvals))
        
        yvals = [np.mean(df_sub.loc[val,limb_str].values)*Config.forceplate_config['fore_hind_post_cm']/2 for key,val in group_row_ids.items()]
        if '2023' in yyyymmdd:
            ax.plot(xvals, 
                     yvals, 
                     color=meanclr,  
                     alpha=0.2, 
                     linewidth = 1)
    
    if '2023' in yyyymmdd:
        # p_text = '*' * (mod.loc[:,'Pr(>|t|)'].iloc[1] < FigConfig.p_thresholds).sum()
        # ax.text(163,1.05, f"head height: {p_text}", 
        #         ha = 'center', fontsize=5)
        
        ax.text(161,1.03,"MSA-def", 
                fontsize=5,color=meanclr)
        ax.hlines(1,161,180,color=meanclr, linestyle='solid', lw=1.5)
    else:
        ax.text(140,1.03,"CTRL", 
                fontsize=5,color=meanclr)
        ax.hlines(1,140,152,color=meanclr, linestyle='dashed', lw=1.5)

    
    # ADD MEANS
    
    # x_centred = np.asarray(df['param'])-np.nanmean(df['param'])
    x_pred = np.linspace(np.nanmin(minmaxs[0])-np.nanmean(df['param']), np.nanmax(minmaxs[1])-np.nanmean(df['param']), endpoint=True)
    
    if mod_type == 'linear':
        y_pred = (mod.loc['(Intercept)', 'Estimate'] +\
                 (mod.loc['param_centred', 'Estimate'] * x_pred) +\
                      np.nanmean(df[limb_str]))*Config.forceplate_config['fore_hind_post_cm']/2
    else:
        y_pred = (mod.loc['(Intercept)', 'Estimate'] +\
                 (mod.loc['poly(param_centred, 2, raw = TRUE)1', 'Estimate'] * x_pred) +\
                 (mod.loc['poly(param_centred, 2, raw = TRUE)2', 'Estimate'] * x_pred**2) +\
                 np.nanmean(df[limb_str]))*Config.forceplate_config['fore_hind_post_cm']/2
   
    x_pred += np.nanmean(df['param'])
    ax.plot(x_pred, 
              y_pred, 
              linewidth=1.5, 
              color=meanclr,
              ls=lnst)
    
############# ADD STATS ############# 
statspath = f"C:\\Users\\MurrayLab\\Documents\\Forceplate\\{yyyymmdds[0]}_x_{yyyymmdds[1]}_mixedEffectsModel_{mod_type}_{variable_str}_v_sBA_egr3_vglut2.csv"
stats = pd.read_csv(statspath, index_col =0)
ax.text(153, 1.03, "vs")

# ptext = "intercept: "
# for j, statscol in enumerate(['trialtypevglut2', 'param_centred:trialtypevglut2']):
#     p = stats.loc[statscol, "Pr(>|t|)"]
#     if (p < np.asarray(FigConfig.p_thresholds)).sum() == 0:
#         ptext += "n.s."    
#     else:
#         ptext += ('*' * (p < np.asarray(FigConfig.p_thresholds)).sum())
#     if j==0:
#         ptext += "; slopes: "
# ax.text(138,
#         0.85,
#         ptext, 
#         fontsize=5)
      
ptext = "intercept: "
for j, (ptext, statscol) in enumerate(zip(
        ['intercept: ', 'slopes: '],
        ['trialtypevglut2', 'param_centred:trialtypevglut2'],
        )):
    p = stats.loc[statscol, "Pr(>|t|)"]
    if (p < np.asarray(FigConfig.p_thresholds)).sum() == 0:
        ptext += "n.s."    
    else:
        ptext += ('*' * (p < np.asarray(FigConfig.p_thresholds)).sum())
    ax.text(160,
            0.78-(j*0.2),
            ptext, ha='center',
            fontsize=5,
            color=FigConfig.colour_config['homolateral'][0])

############# ADD STATS ############# 

        
ax.set_xlabel('Snout-hump angle\n(deg)')

ax.set_xticks([140,160,180])
ax.set_xlim(135,180)
ax.set_yticks([-1.0,-0.5,0,0.5,1.0])
ax.set_ylim(-1.2,1)
ax.set_title('Head height trials')

ax.set_ylabel("AP centre of support\n(cm)")
# axes[2].set_ylabel("Total detected weight (%)")
plt.tight_layout()
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplateEGR3_CoMy_{param}_{limb_str}.svg'),
            transparent = True,
            dpi = 300)
