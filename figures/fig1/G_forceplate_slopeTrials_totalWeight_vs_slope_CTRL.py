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
                                               appdx = f'_{param}')

mice = np.unique(df['mouse'])
fig, ax = plt.subplots(1, 1, figsize=(1.34, 1.39))
ax.hlines(100,xmin=-40,xmax=40, ls = 'dashed', color = 'grey')

limb_clr = 'greys'
limb_str = 'headplate_weight_frac'
variable_str = 'headWfrac'

# CONVERT HEADPLATE VALUES TO PERCENTAGE OF DETECTED WEIGHT
df[limb_str] = -(df[limb_str]-1)*100

starts = np.empty(len(mice)); ends = np.empty(len(mice))
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    headHWs = np.unique(df_sub['param'])

    yvals = [np.nanmean(df_sub[df_sub['param']==h][limb_str]) for h in np.unique(df_sub['param'])]
    starts[im] = yvals[0]; ends[im] = yvals[-1]
    ax.plot(headHWs, 
             yvals, 
             color=FigConfig.colour_config[limb_clr][1],  
             alpha=0.4, 
             linewidth = 0.7)

print(f"Max decline: {np.mean(starts):.3f} ± {scipy.stats.sem(starts):.3f}")
print(f"Max incline: {np.mean(ends):.3f} ± {scipy.stats.sem(ends):.3f}")

# APPROXIMATE WITH A FUNCTION
from scipy.optimize import curve_fit
from scipy.stats import t
def linear_fit(x,A,B):
    return A + B * x

x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)

popt,pcov = curve_fit(linear_fit, df['param'].values, df[limb_str].values, p0=(0.5,0))
A_fit, B_fit = popt
print(f"Linear fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}")
ax.plot(x_pred, 
              linear_fit(x_pred, *popt), 
              linewidth=1.5, 
              color=FigConfig.colour_config[limb_clr][1])
std_err = np.sqrt(np.diag(pcov)) # standard errors
t_values = popt/std_err
dof = max(0, len(df[limb_str].values)-len(popt))   
p_values = [2 * (1 - t.cdf(np.abs(t_val), dof)) for t_val in t_values]
print(f"p-values: A_p = {p_values[0]:.3e}, B_p = {p_values[1]:.3e}")
for i_p, (p, exp_d_param) in enumerate(zip(
        p_values[1:], 
        ["slope"],
        )):
    p_text = ('*' * (p < FigConfig.p_thresholds).sum())
    if (p < FigConfig.p_thresholds).sum() == 0:
        p_text += "n.s."
    ax.text(0,
                 175, 
                 f"{exp_d_param}: {p_text}", 
                 ha = 'center', 
                 color = FigConfig.colour_config[limb_clr][1],
                 fontsize = 5)

# if A is significant, it means that the data asymptotes at a non-zero value
# if B is significant, it means that there is a (monotonic?) change in y as the x changes
# if k is significant, it means that y approaches the asymptote in an exponential manner

        
ax.set_xlabel('Surface slope (deg)')

ax.set_xticks([-40,-20,0,20,40][::2])
ax.set_xlim(-50,50)
ax.set_yticks([60,100,140,180])
ax.set_ylim(40,180)
ax.set_title("Surface slope trials")


ax.set_ylabel("Total leg load (%)")
plt.tight_layout()

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_totalWeight_{param}_{limb_str}.svg'),
        transparent = True,
        dpi = 300)
