import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\thesis")

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

yyyymmdd = '2021-10-26'
param = 'headHW'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_'+param)

mice = np.unique(df['mouse'])

fig, axes = plt.subplots(1,1, figsize=(1.45, 1.47))

variable = 'CoMx_mean'
clr = 'greys'
variable_str = 'CoMx'
    
diffs = np.empty(len(mice))
stds = np.empty((len(mice), 8))*np.nan
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    headHWs = np.unique(df_sub['param'])
    
    yvals = [np.nanmean(df_sub[df_sub['param']==h][variable]*Config.forceplate_config['fore_hind_post_cm']/2) for h in np.unique(df_sub['param'])]
    
    # quantify variability
    stds[im, :len(headHWs)] = [np.nanstd(df_sub[df_sub['param']==h][variable]*Config.forceplate_config['fore_hind_post_cm']/2) for h in np.unique(df_sub['param'])]
    
    diffs[im] = yvals[-1]-yvals[0]
    axes.plot(headHWs, 
             yvals, 
             color=FigConfig.colour_config[clr][1],  
             alpha=0.4, 
             linewidth = 0.7)
print(f"Average standard deviation: {np.mean(np.nanmean(stds, axis=1))}")
print(f"Mean change over the examined x range: {np.mean(diffs):.3f} ± {scipy.stats.sem(diffs):.3f}")

# LOAD MIXED-EFFECTS MODEL
slope_enforced = 'slopeENFORCED'
mod = 'Slope'
path = f"{Config.paths['forceplate_output_folder']}\\{yyyymmdd}_mixedEffectsModel_linear_{variable_str}_{param}_{slope_enforced}_rand{mod}.csv"
stats_df = pd.read_csv(path, index_col=0)

# A + Bx
x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)
x_centred = x_pred - np.nanmean(df['param'].values)
y_centred = stats_df.loc['(Intercept)', 'Estimate'] + (stats_df.loc['indep_var_centred', 'Estimate'] * x_centred)
y_pred = y_centred + np.nanmean(df[variable].values)

axes.plot(x_pred, 
              y_pred, 
              linewidth=1.5, 
              color=FigConfig.colour_config[clr][1])

# PLOT STATS
t = stats_df.loc['indep_var_centred', 't value']
p = stats_df.loc['indep_var_centred', 'Pr(>|t|)']
print(f"{variable}: mean={stats_df.loc['indep_var_centred', 'Estimate']:.4g}, SEM={stats_df.loc['indep_var_centred', 'Std. Error']:.4g}, t={t:.3f}, p={p:.3g}")
p_text = "n.s." if (p < FigConfig.p_thresholds).sum() == 0 else ('*' * (p < FigConfig.p_thresholds).sum())

axes.text(0.6,
             0.9, 
             f"slope: {p_text}", 
             ha = 'center', 
             color = FigConfig.colour_config[clr][1],
             fontsize = 5)

axes.set_xlabel('Weight-adjusted\nhead height')
axes.set_xticks([0,0.6,1.2])
axes.set_xlim(0,1.2)
axes.set_yticks([-1.5,-1.0,-0.5,0,0.5,1.0])
axes.set_ylim(-1.5,1.0)
axes.set_title("Head height trials")

axes.set_ylabel("Mediolateral\nCoS (cm)")
plt.tight_layout()
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_CoMx_vs_headHW.svg'),
            transparent = True,
            dpi =300)


