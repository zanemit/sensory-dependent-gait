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

limb_clr = 'greys'
limb_str = 'headplate_weight_frac'

# CONVERT HEADPLATE VALUES TO PERCENTAGE OF DETECTED WEIGHT
df[limb_str] = -(df[limb_str]-1)*100

group_num =5
slopes = []
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    df_sub['param_bins'] = pd.qcut(df_sub['param'], q=group_num, labels=False)
    
    yvals = df_sub.groupby('param_bins')[limb_str].mean().values
    
    bin_edges = pd.qcut(df_sub['param'], q=group_num).cat.categories
    sbas = [(bin_edges[i].left + bin_edges[i].right) / 2 for i in range(len(bin_edges))]
    ax.plot(sbas, 
             yvals, 
             color=FigConfig.colour_config[limb_clr][1],  
             alpha=0.4, 
             linewidth = 0.7)
    popt_m,pcov_m = curve_fit(linear_fit, sbas, yvals, p0=(0.5,0))
    slopes.append(popt_m[1])
print(f"Average slope across mice: {np.mean(slopes)} ± {np.std(slopes)/np.sqrt(len(slopes))}")

# APPROXIMATE WITH A FUNCTION
x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)
popt,pcov = curve_fit(linear_fit, df['param'].values, df[limb_str].values, p0=(0.5,0))
A_fit, B_fit = popt
print(f"Linear fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}")
ax.plot(x_pred, 
              linear_fit(x_pred, *popt), 
              linewidth=1.5, 
              color=FigConfig.colour_config[limb_clr][2])
# print(f"LAST VALUE: {exp_decay(x_pred, *popt)[-1]}")
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
    ax.text(160,
                 185-(i_p*15), 
                 f"{exp_d_param}: {p_text}", 
                 ha = 'center', 
                 color = FigConfig.colour_config[limb_clr][1],
                 fontsize = 5)

# mixed-effects stats
# stats_path = r"C:\Users\MurrayLab\Documents\Forceplate\2023-11-06_mixedEffectsModel_linear_headWfrac_snoutBodyAngle.csv"
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

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_totalWeight_{param}_{limb_str}.svg'),
        transparent = True,
        dpi = 300)
