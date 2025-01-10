import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
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
                                               appdx = f'_{param}')

mice = np.unique(df['mouse'])
fig, ax = plt.subplots(1, 1, figsize=(1.33, 1.47))
ax.hlines(100,xmin=0,xmax=1.2, ls = 'dashed', color = 'grey')

limb_clr = 'greys'
limb_str = 'headplate_weight_frac'

# CONVERT HEADPLATE VALUES TO PERCENTAGE OF DETECTED WEIGHT
df[limb_str] = -(df[limb_str]-1)*100

for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    headHWs = np.unique(df_sub['param'])
    
    yvals = [np.nanmean(df_sub[df_sub['param']==h][limb_str]) for h in np.unique(df_sub['param'])]
    ax.plot(headHWs, 
             yvals, 
             color=FigConfig.colour_config[limb_clr][1],  
             alpha=0.4, 
             linewidth = 0.7)


# APPROXIMATE WITH A FUNCTION
from scipy.optimize import curve_fit
from scipy.stats import t
def exp_decay(x,A,B,k):
    return A - B * np.exp(-k*x)

x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)
popt,pcov = curve_fit(exp_decay, df['param'].values, df[limb_str].values, p0=(np.nanmax(df[limb_str].values),
                                                                              np.nanmax(df[limb_str].values)-np.nanmin(df[limb_str].values),
                                                                              1/np.nanmean(df['param'].values)))
A_fit, B_fit, k_fit = popt
print(f"Exp decay fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}, k = {k_fit:.3f}")
ax.plot(x_pred, 
              exp_decay(x_pred, *popt), 
              linewidth=1.5, 
              color=FigConfig.colour_config[limb_clr][1])
# print(f"LAST VALUE: {exp_decay(x_pred, *popt)[-1]}")
std_err = np.sqrt(np.diag(pcov)) # standard errors
t_values = popt/std_err
dof = max(0, len(df[limb_str].values)-len(popt))   
p_values = [2 * (1 - t.cdf(np.abs(t_val), dof)) for t_val in t_values]
print(f"p-values: A_p = {p_values[0]:.3e}, B_p = {p_values[1]:.3e}, k_p = {p_values[2]:.3e}")
for i_p, (p, exp_d_param) in enumerate(zip(
        p_values[1:], 
        ["scale factor", "rate constant"],
        )):
    p_text = ('*' * (p < FigConfig.p_thresholds).sum())
    if (p < FigConfig.p_thresholds).sum() == 0:
        p_text += "n.s."
    ax.text(0.6,
                 170-(i_p*15), 
                 f"{exp_d_param}: {p_text}", 
                 ha = 'center', 
                 color = FigConfig.colour_config[limb_clr][1],
                 fontsize = 5)


# if A is significant, it means that the data asymptotes at a non-zero value
# if B is significant, it means that there is a (monotonic?) change in y as the x changes
# if k is significant, it means that y approaches the asymptote in an exponential manner

        
ax.set_xlabel('Weight-adjusted\nhead height')

ax.set_xticks([0,0.6,1.2])
ax.set_xlim(-0.1,1.2)
ax.set_yticks([40,70,100,130,160])
ax.set_ylim(30,160)
ax.set_title("Head height trials")


ax.set_ylabel("Total leg load (%)")
plt.tight_layout()

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_totalWeight_{param}_{limb_str}.svg'),
        transparent = True,
        dpi = 300)
