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

from scipy.optimize import curve_fit
from scipy.stats import t
def linear_fit(x,A,B):
    return A + B * x

yyyymmdd = '2021-10-26'
param = 'snoutBodyAngle'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = f'_{param}')

mice = np.unique(df['mouse'])
fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.47))
ax.hlines(100,xmin=135,xmax=180, ls = 'dashed', color = 'grey')

clr = 'greys'
variable = 'headWfrac'
variable_str = 'headplate_weight_frac'

# CONVERT HEADPLATE VALUES TO PERCENTAGE OF DETECTED WEIGHT
df[variable_str] = -(df[variable_str]-1)*100

group_num =5
slopes = []
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m].copy()
    df_sub.loc[:,'param_bins'] = pd.qcut(df_sub['param'], q=group_num, labels=False)
    
    yvals = df_sub.groupby('param_bins')[variable_str].mean().values
    
    bin_edges = pd.qcut(df_sub['param'], q=group_num).cat.categories
    sbas = [(bin_edges[i].left + bin_edges[i].right) / 2 for i in range(len(bin_edges))]
    ax.plot(sbas, 
             yvals, 
             color=FigConfig.colour_config[clr][1],  
             alpha=0.4, 
             linewidth = 0.7)
    popt_m,pcov_m = curve_fit(linear_fit, sbas, yvals, p0=(0.5,0))
    slopes.append(popt_m[1])
print(f"Average slope across mice: {np.mean(slopes)} ± {np.std(slopes)/np.sqrt(len(slopes))}")

# LOAD MIXED-EFFECTS MODEL
slope_enforced = 'slopeENFORCED'
mod = 'Slope'
path = f"{Config.paths['forceplate_output_folder']}\\{yyyymmdd}_mixedEffectsModel_linear_{variable}_{param}_{slope_enforced}_rand{mod}.csv"
stats_df = pd.read_csv(path, index_col=0)

# A + Bx
x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)
x_centred = x_pred - np.nanmean(df['param'].values)
y_centred = stats_df.loc['(Intercept)', 'Estimate'] + (stats_df.loc['indep_var_centred', 'Estimate'] * x_centred)
y_pred = y_centred + np.nanmean(df[variable_str].values)

ax.plot(x_pred, 
        y_pred, 
        linewidth=1.5, 
        color=FigConfig.colour_config[clr][1])

# PLOT STATS
t = stats_df.loc['indep_var_centred', 't value']
p = stats_df.loc['indep_var_centred', 'Pr(>|t|)']
print(f"{variable_str}: mean={stats_df.loc['indep_var_centred', 'Estimate']:.4g}, SEM={stats_df.loc['indep_var_centred', 'Std. Error']:.4g}, t={t:.3f}, p={p:.3g}")
p_text = "n.s." if (p < FigConfig.p_thresholds).sum() == 0 else ('*' * (p < FigConfig.p_thresholds).sum())

ax.text(160,175, f"slope: {p_text}", ha = 'center', 
        color = FigConfig.colour_config[clr][1], 
        fontsize = 5)

# if A is significant, it means that the data asymptotes at a non-zero value
# if B is significant, it means that there is a (monotonic?) change in y as the x changes
# if k is significant, it means that y approaches the asymptote in an exponential manner

        
ax.set_xlabel('Snout-hump angle\n(deg)')

ax.set_xticks([140,160,180])
ax.set_xlim(135,180)
ax.set_yticks([60,100,140,180])
ax.set_ylim(40,180)
ax.set_title("Head height trials")

ax.set_ylabel("Total leg load (%)")
plt.tight_layout()

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_totalWeight_{param}_{variable_str}.svg'),
        transparent = True,
        dpi = 300)
