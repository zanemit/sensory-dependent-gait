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

modQDR = pd.read_csv(Path(Config.paths["forceplate_output_folder"]) / f"{yyyymmdd}_mixedEffectsModel_linear_{variable_str}_{param}.csv", index_col=0)

x_centered = df['param'] - np.nanmean(df['param'])
x_pred = np.linspace(np.nanmin(x_centered), np.nanmax(x_centered), endpoint=True)


# APPROXIMATE WITH A FUNCTION
from scipy.optimize import curve_fit
from scipy.stats import t
def linear_fit(x,A,B):
    return A + B * x

x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)

popt,pcov = curve_fit(linear_fit, df['param'].values, df[variable].values, p0=(0,0))
A_fit, B_fit = popt
print(f"Linear fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}")
axes.plot(x_pred, 
              linear_fit(x_pred, *popt), 
              linewidth=1.5, 
              color=FigConfig.colour_config[clr][1])
# std_err = np.sqrt(np.diag(pcov)) # standard errors
# t_values = popt/std_err
# dof = max(0, len(df[variable].values)-len(popt))   
# p_values = [2 * (1 - t.cdf(np.abs(t_val), dof)) for t_val in t_values]
p_value = modQDR.loc['param_centred', 'Pr(>|t|)']
print(f"p-value: {p_value}")

p_text = ('*' * (p_value < FigConfig.p_thresholds).sum())
if (p_value < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
axes.text(0.6,
             0.9, 
             f"slope: {p_text}", 
             ha = 'center', 
             color = FigConfig.colour_config[clr][2],
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


