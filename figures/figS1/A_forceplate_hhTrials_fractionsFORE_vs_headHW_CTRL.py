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
from scipy.stats import wilcoxon
def exp_decay(x,A,B,k):
    return A - B * np.exp(-k*x)
def linear_fit(x,A,B):
    return A + B * x

x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)

# TOTAL FOR PLOTTING
popt,pcov = curve_fit(exp_decay, df['param'].values, df[limb_str].values, p0=(0.25,0.75,2))
  
A_fit, B_fit, k_fit = popt
print(f"Exp decay fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}, k = {k_fit:.3f}")
y_pred = exp_decay(x_pred, *popt)
axes.plot(x_pred, 
              y_pred, 
              linewidth=1.5, 
              color=FigConfig.colour_config[limb_clr][2])

# STATS PER SUBJECT
p_list = []
for m in df['mouse'].unique():
    df_sub = df[df['mouse']==m].copy()
    popt, pcov = curve_fit(exp_decay, df_sub['param'].values, df_sub[limb_str].values, p0=(A_fit, B_fit, k_fit), maxfev=10000)
    p_list.append(popt)
    
params = np.array(p_list) #(n_subjects, n_params)
param_names = ['A', 'B', 'k']
params_df = pd.DataFrame(params, columns=param_names, index=df['mouse'].unique())

# STATS ACROSS SUBJECTS
summary = []
for col in param_names:
    vals = params_df[col].dropna().values
    median = np.median(vals)
    w_stat, p = wilcoxon(vals)
    summary.append((col, median, w_stat, p))
for col, median,  w_stat, p in summary:
    print(f"{col}: median={median:.4g}, w({params_df.shape[0]-1})={w_stat:.3f}, p={p:.3g}")
    
# PLOT STATS
for i_p, (p_k, exp_d_param) in enumerate(zip(
        [summary[1][3], summary[2][3]],
        ['scale factor', 'rate constant']
        )):
    p_text = "n.s." if (p_k < FigConfig.p_thresholds).sum() == 0 else ('*' * (p_k < FigConfig.p_thresholds).sum())

    axes.text(0.63,
              1.46-(i_p*0.15), 
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
