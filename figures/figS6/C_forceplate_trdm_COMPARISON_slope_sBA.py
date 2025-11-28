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
dep_var = 'snoutBodyAngle'
fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.4))
bin_edges = np.linspace(-40.0001,40, group_num+1, endpoint=True)

from scipy.optimize import curve_fit
from scipy.stats import wilcoxon
def linear(x,A,B):
    return A - B*x

x_data_comb = np.empty(0)
y_data_comb = np.empty(0)
n_data = []
ssrs = []
B_FP = []; B_TRDM = []
for i, (yyyymmdd, otp_dir, appdx, data_str, indep_var, clr, lnst, lbl, B_list) in enumerate(zip(
                                ['2022-04-04', '2022-08-18'], 
                                [Config.paths["forceplate_output_folder"], Config.paths["passiveOpto_output_folder"]],
                                ['_levels', '_incline'],
                                ['forceplateAngleParams', 'bodyAngles'],
                                ['headLVL', 'incline'],
                                [FigConfig.colour_config['headbars'], FigConfig.colour_config['homolateral'][2]], 
                                ['dashed', 'solid'],
                                ['Force sensors', 'Treadmill'],
                                [B_FP, B_TRDM]
                                )):
    df, _ = data_loader.load_processed_data(outputDir = otp_dir, 
                                                   dataToLoad=data_str, 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = appdx)
    
    
    # APPROXIMATE WITH A FUNCTION
       
    if 'Passive' in otp_dir: # preOpto
        df = df.loc[:200,(slice(None),slice(None),slice(None), slice(None), 'snoutBody')]
    
    if indep_var == 'incline':
        df = pd.DataFrame({'incline': [-int(x[3:]) for x in df.columns.get_level_values(3)],
                           'snoutBodyAngle': df.values.mean(axis=0),
                           'mouseID': df.columns.get_level_values(0)})
    df['indep_bins'], bin_edges = pd.cut(df[indep_var], bins=bin_edges, retbins = True, labels = False)

    summary = df.groupby('indep_bins')[dep_var].agg(['std', 'sem']).reset_index()
    summary['bin_x'] = [np.mean((bin_edges[i],bin_edges[i+1])) for i in range(bin_edges.shape[0]-1)]
    
    x_pred = np.linspace(summary['bin_x'].min(), summary['bin_x'].max(), endpoint=True)
    mask = ~np.isnan(df[dep_var].values)
    popt,pcov = curve_fit(linear, df[indep_var].values[mask], df[dep_var].values[mask], p0=(np.nanmean(df[dep_var].values),
                                                                                  (np.nanmax(df[dep_var].values)-np.nanmin(df[dep_var].values))/(np.nanmax(df[indep_var].values)-np.nanmin(df[indep_var].values)),
                                                                                  ))
    A_fit, B_fit = popt
    print(f"Linear fitted params (total): A = {A_fit:.3f}, B = {B_fit:.3f}")
    y_pred = linear(x_pred, *popt)
    
    ids = np.linspace(0, len(x_pred)-1, group_num, dtype=int)
   
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
        edgecolor = None,
        )
    
    ax.plot(x_pred,#+(i*0.05), 
                  y_pred, 
                  linewidth=1.5, 
                  linestyle = lnst,
                  color=clr,
                  label = lbl)
    
    # plot horizontal line as a legend
    ax.hlines(178.5, 6 - (i*50), 38 - (i*46), color = clr, ls = lnst, lw = 0.7)
    
    # COMBINE DATA
    x_data_comb = np.concatenate((x_data_comb, df[indep_var].values[mask]))
    y_data_comb = np.concatenate((y_data_comb, df[dep_var].values[mask]))
    n_data.append(df[indep_var].values[mask].shape[0])
    ssrs.append(np.sum((df[dep_var].values[mask] - linear(df[indep_var].values[mask], *popt))**2))
    
    # FIT LINES PER MOUSE
    p_list=[]
    for m in df['mouseID'].unique():
        df_sub = df[df['mouseID']==m].copy()
        mask_m = ~np.isnan(df_sub[dep_var].values)
        popt_m, pcov_m = curve_fit(linear, df_sub[indep_var].values[mask_m], 
                                   df_sub[dep_var].values[mask_m], p0=(A_fit, B_fit), maxfev=10000)
        B_list.append(popt_m[1])
        p_list.append(popt_m)
    
    params = np.array(p_list) #(n_subjects, n_params)
    param_names = ['A', 'B']
    params_df = pd.DataFrame(params, columns=param_names, index=df['mouseID'].unique())

    # STATS ACROSS SUBJECTS
    summary = []
    for col in param_names:
        vals = params_df[col].dropna().values
        median = np.median(vals)
        w_stat, p = wilcoxon(vals)
        summary.append((col, median, w_stat, p))
    for col, median,  w_stat, p in summary:
        print(f"{data_str} {col}: median={median:.4g}, w({params_df.shape[0]-1})={w_stat:.3f}, p={p:.3g}")
    print("\n\n")

ax.set_xlim(-50,50)
ax.set_ylim(135,180)
ax.set_ylabel('Snout-body angle (deg)')
ax.set_xlabel('Surface slope (deg)')
ax.set_xticks([-40,-20,0,20,40], labels = [-40,-20,0,20,40])
ax.set_yticks([140,150,160,170,180])

plt.tight_layout()


# COMPUTE WILCOXON RANK SUM TEST STATISTIC
from scipy.stats import mannwhitneyu

stat, p_value = mannwhitneyu(B_FP, B_TRDM)
print(stat, p_value)

ax.text(-45,179, "treadmill ", ha = 'left', 
        color = FigConfig.colour_config['homolateral'][2], 
        fontsize = 5)
p_text = "n.s." if (p_value < FigConfig.p_thresholds).sum() == 0 else ('*' * (p_value < FigConfig.p_thresholds).sum())
p_text = "vs sensors: " + p_text
ax.text(-6,179, p_text, ha = 'left', 
        color = FigConfig.colour_config['headbars'], 
        fontsize = 5)

ax.set_title("Surface slope trials")
plt.tight_layout()

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"forceplate_snoutBodyAngles_vs_slope_COMPARISON.svg",
            dpi=300,
            transparent=True)



