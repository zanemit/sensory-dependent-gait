import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

group_num = 5
indep_var = 'headHW'
dep_var = 'snoutBodyAngle'
fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.47))
bin_edges = np.linspace(0,1.2, group_num+1, endpoint=True)

yyyymmdd = '2022-08-18'
df, _ = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                               dataToLoad="locomParams", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_FPcomparison')

x_data_comb = np.empty(0)
y_data_comb = np.empty(0)
n_data = []
ssrs = []
for i, (setup, clr, lnst, lbl) in enumerate(zip(['FP', 'TRDMstat'], 
                                                [FigConfig.colour_config['headbars'], FigConfig.colour_config['homolateral'][2]], 
                                                ['dashed', 'solid'],
                                                ['Force sensors', 'Treadmill'])):
    df_sub = df[df['setup'] == setup]
    
    # APPROXIMATE WITH A FUNCTION
    from scipy.optimize import curve_fit
    from scipy.stats import t
    def exp_decay(x,A,B,k):
        return A - B * np.exp(-k*x)
    
    df_sub['indep_bins'], bin_edges = pd.qcut(df_sub[indep_var], group_num, retbins = True, labels = False)
    # df_sub['indep_bins'] = pd.cut(df_sub[indep_var], bins = bin_edges, labels = False)
    summary = df_sub.groupby('indep_bins')[dep_var].agg(['std', 'sem']).reset_index()
    summary['bin_x'] = [np.mean((bin_edges[i],bin_edges[i+1])) for i in range(bin_edges.shape[0]-1)]
    
    x_pred = np.linspace(summary['bin_x'].min(), summary['bin_x'].max(), endpoint=True)
    mask = ~np.isnan(df_sub[dep_var].values)
    popt,pcov = curve_fit(exp_decay, df_sub[indep_var].values[mask], df_sub[dep_var].values[mask], p0=(np.nanmax(df_sub[dep_var].values),
                                                                                  np.nanmax(df_sub[dep_var].values)-np.nanmin(df_sub[dep_var].values),
                                                                                  1/np.nanmean(df_sub[indep_var].values)))
    A_fit, B_fit, k_fit = popt
    print(f"Exp decay fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}, k = {k_fit:.3f}")
    y_pred = exp_decay(x_pred, *popt)
    
    ids = np.linspace(0, len(x_pred)-1, group_num, dtype=int)
    # for k in range(df_sub['indep_bins'].max()+1):
    #     df_sub_k = df_sub[df_sub['indep_bins'] == k][dep_var].values
    #     ax.boxplot(df_sub_k[~np.isnan(df_sub_k)],
    #                positions = [x_pred[ids][k]+(i*0.05)],
    #                widths = 0.03,
    #                medianprops = dict(color = clr, linewidth = 1, alpha = 0.4),
    #                            boxprops = dict(color = clr, linewidth = 1, alpha = 0.4), capprops = dict(color = clr, linewidth = 1, alpha = 0.4),
    #                            whiskerprops = dict(color = clr, linewidth = 1, alpha = 0.4), flierprops = dict(mec = clr, linewidth = 1, alpha = 0.4, ms=2))
    # 95% confidence intervals
    ids = np.linspace(0, len(x_pred)-1, group_num, dtype=int)
    summary['ci95_hi'] = y_pred[ids] + summary['std']  + 1.96*summary['sem']
    summary['ci95_lo'] = y_pred[ids] - summary['std'] - 1.96*summary['sem']
    ax.fill_between(
        x_pred[ids],
        summary['ci95_lo'],
        summary['ci95_hi'],
        alpha = 0.3,
        color = clr,
        edgecolor = None
        )
    
    ax.plot(x_pred,#+(i*0.05), 
                  y_pred, 
                  linewidth=1.5, 
                  linestyle = lnst,
                  color=clr,
                  label = lbl)
    
    ax.set_xlim(0,1.2)
    ax.set_ylim(135,180)
    ax.set_ylabel('Snout-body angle (deg)')
    ax.set_xlabel('Weight-adjusted\nhead height')
    ax.set_xticks([0,0.4,0.8,1.2], labels = [0,0.4,0.8,1.2])
    ax.set_yticks([140,150,160,170,180])
    
    # plot horizontal line as a legend
    ax.hlines(178, 0.7 - (i*0.66), 1.12 - (i*0.63), color = clr, ls = lnst, lw = 0.7)
    
    # COMBINE DATA
    x_data_comb = np.concatenate((x_data_comb, df_sub[indep_var].values[mask]))
    y_data_comb = np.concatenate((y_data_comb, df_sub[dep_var].values[mask]))
    n_data.append(df_sub[indep_var].values[mask].shape[0])
    ssrs.append(np.sum((df_sub[dep_var].values[mask] - exp_decay(df_sub[indep_var].values[mask], *popt))**2))

plt.tight_layout()

# COMPUTE F STATISTIC
def log_likelihood(ssr,n):
    return -n/2 * (np.log(2 * np.pi * ssr/n)+1)
n_data1, n_data2 = n_data
ssr1, ssr2 = ssrs
logL1 = log_likelihood(ssr1, n_data1)
logL2 = log_likelihood(ssr2, n_data2)

from scipy.stats import chi2
logL_comb = log_likelihood(ssr1+ssr2, n_data1+n_data2)
lr_stat = -2 * (logL_comb - (logL1+logL2))
dof = len(popt)
p_value = chi2.sf(lr_stat,dof)
print(f"Comparison p-value: {p_value}")

ax.text(0.02,179, "treadmill ", ha = 'left', 
        color = FigConfig.colour_config['homolateral'][2], 
        fontsize = 5)
p_text = "vs sensors: " + ('*' * (p_value < FigConfig.p_thresholds).sum())
if (p_value < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
ax.text(0.55,179, p_text, ha = 'left', 
        color = FigConfig.colour_config['headbars'], 
        fontsize = 5)
ax.set_title("Head height trials")
plt.tight_layout()

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"forceplate_snoutBodyAngles_vs_headHW_COMPARISON.svg",
            dpi=300,
            transparent=True)



