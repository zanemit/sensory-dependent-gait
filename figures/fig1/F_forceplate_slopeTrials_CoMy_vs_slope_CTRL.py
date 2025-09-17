import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

yyyymmdd = '2022-04-04'
param = 'levels'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_'+param)

mice = np.unique(df['mouse'])

fig, ax = plt.subplots(1,1, figsize=(1.45, 1.39))

variable = 'CoMy_mean'
clr = 'homolateral'
variable_str = 'CoMy'
tlt = 'Surface slope trials'

stds1 = np.empty((len(mice), 8))*np.nan

diffs = np.empty(len(mice))
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    headHWs = np.unique(df_sub['param'])
    
    yvals = [np.nanmean(df_sub[df_sub['param']==h][variable]*Config.forceplate_config['fore_hind_post_cm']/2) for h in np.unique(df_sub['param'])]
    # quantify variability
    stds1[im, :len(headHWs)] = [np.nanstd(df_sub[df_sub['param']==h][variable]*Config.forceplate_config['fore_hind_post_cm']/2) for h in np.unique(df_sub['param'])]
    
    
    diffs[im] = yvals[-1]-yvals[0]
    ax.plot(headHWs, 
             yvals, 
             color=FigConfig.colour_config[clr][2],  
             alpha=0.4, 
             linewidth = 0.7)
    
print(f"Average standard deviation: {np.mean(np.nanmean(stds1, axis=1))}")

modLIN = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_linear_BEST_{variable_str}_{param}.csv")

x_centred = np.asarray(df['param'])-np.nanmean(df['param'])
x_pred = np.linspace(x_centred.min(), x_centred.max(), 10, endpoint=True)
# ax.set_xlim(np.nanmin(minmaxs[0])-(0.1*(np.nanmax(minmaxs[1])-np.nanmin(minmaxs[0]))),
#             np.nanmax(minmaxs[1])+(0.1*(np.nanmax(minmaxs[1])-np.nanmin(minmaxs[0]))))

y_predLIN = (modLIN['Estimate'][0] +\
             modLIN['Estimate'][1] * x_pred +\
            np.nanmean(df[variable])) * Config.forceplate_config['fore_hind_post_cm']/2
x_pred += np.nanmean(df['param'])
ax.plot(x_pred, 
        y_predLIN, 
        linewidth=1.5, 
        color=FigConfig.colour_config[clr][2])

print(f"Intercept: {((modLIN['Estimate'][0] + np.nanmean(df[variable]))* Config.forceplate_config['fore_hind_post_cm']/2):.3f}") 
print(f"Slope: {(modLIN['Estimate'][1]* Config.forceplate_config['fore_hind_post_cm']/2):.3f}") 

print(f"Mean change over 80 degrees: {np.mean(diffs):.3f} ± {scipy.stats.sem(diffs):.3f}")
print(f"{variable} is modulated by {(10*modLIN['Estimate'][1]*Config.forceplate_config['fore_hind_post_cm']*10/2):.1f} ± {(10*modLIN['Std. Error'][1]*Config.forceplate_config['fore_hind_post_cm']*100/2):.1f} mm per 10 deg")

# p values
print(f"p-value: {modLIN['Pr(>|t|)'][1]}")
p_text = "slope: " + ('*' * (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum())
if (modLIN['Pr(>|t|)'][1] < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
ax.text(0,1, p_text, ha = 'center', 
        color = FigConfig.colour_config[clr][2], 
        fontsize = 5)

ax.set_xlabel('Surface slope (deg)')
ax.set_xticks([-40,-20,0,20,40][::2])
ax.set_xlim(-50,50)
ax.set_yticks([-1.0,-0.5,0,0.5,1])
ax.set_ylim(-1.3,1)
ax.set_title(tlt)

ax.set_ylabel("Anteroposterior\nCoS (cm)")
plt.tight_layout()
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_CoMy_vs_incline.svg'),
            transparent = True,
            dpi =300)


