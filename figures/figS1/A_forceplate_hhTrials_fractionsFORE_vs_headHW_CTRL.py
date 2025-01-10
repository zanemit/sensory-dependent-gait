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

yyyymmdd = '2021-10-26'
param = 'headHW'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="meanParamDF", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = f'_{param}')

mice = np.unique(df['mouse'])
fig, axes = plt.subplots(1, 1, figsize=(1.3, 1.47))

limb_clr = 'homologous'
limb_str = 'fore_weight_frac'
tlt = 'Head height trials'
variable_str = 'foreWfrac'
perc_changes = []
for im, m in enumerate(mice):
    df_sub = df[df['mouse'] == m]
    headHWs = np.unique(df_sub['param'])
    
    yvals = [np.nanmean(df_sub[df_sub['param']==h][limb_str]) for h in np.unique(df_sub['param'])]
    axes.plot(headHWs, 
             yvals, 
             color=FigConfig.colour_config[limb_clr][2],  
             alpha=0.4, 
             linewidth = 0.7)
    perc_changes.append((yvals[-1]-yvals[0])*100/yvals[0])
             
# # fore-hind and comxy plot means


# APPROXIMATE WITH A FUNCTION
from scipy.optimize import curve_fit
from scipy.stats import t
def exp_decay(x,A,B,k):
    return A - B * np.exp(-k*x)
def linear_fit(x,A,B):
    return A + B * x

x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)


popt,pcov = curve_fit(exp_decay, df['param'].values, df[limb_str].values, p0=(0.25,0.75,2))
# popt,pcov = curve_fit(exp_decay, df['param'].values, df[limb_str].values, p0=(np.nanmax(df[limb_str].values),
#                                                                               np.nanmax(df[limb_str].values)-np.nanmin(df[limb_str].values),
#                                                                               1/np.nanmean(df['param'].values)))
A_fit, B_fit, k_fit = popt
print(f"Exp decay fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}, k = {k_fit:.3f}")
y_pred = exp_decay(x_pred, *popt)
axes.plot(x_pred, 
              y_pred, 
              linewidth=1.5, 
              color=FigConfig.colour_config[limb_clr][2])
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
    axes.text(0.63,
                 1.46-(i_p*0.2), 
                 f"{exp_d_param}: {p_text}", 
                 ha = 'center', 
                 color = FigConfig.colour_config[limb_clr][2],
                 fontsize = 5)

print(f"Forelimb load changed by {np.mean(perc_changes)} ± {np.std(perc_changes)/np.sqrt(len(perc_changes))}%")
# if A is significant, it means that the data asymptotes at a non-zero value
# if B is significant, it means that there is a (monotonic?) change in y as the x changes
# if k is significant, it means that y approaches the asymptote in an exponential manner

        
axes.set_xlabel('Weight-adjusted\nhead height')

axes.set_xticks([0,0.6,1.2])
axes.set_xlim(0,1.2)
axes.set_yticks([0,0.5,1.0,1.5])
axes.set_ylim(-0.1,1.5)
axes.set_title(tlt)

axes.set_ylabel("Forelimb weight fraction")
# axes[2].set_ylabel("Total detected weight (%)")
plt.tight_layout()
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_foreHeadWeights_{param}_{limb_str}.svg'),
            transparent = True,
            dpi = 300)
