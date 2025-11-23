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
dep_var = 'rH1x'
fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.4), sharex = True)
bin_edges = np.linspace(-40.0001,40, group_num+1, endpoint=True)

x_data_comb = np.empty(0)
y_data_comb = np.empty(0)
n_data = []
ssrs = []
B_FP = []; B_TRDM = []
for i, (yyyymmdd, otp_dir, appdx, indep_var, appdx2, clr, lnst, lbl, cfg, B_list) in enumerate(zip(
                                ['2022-04-04', '2022-08-18'], # yyyymmdd
                                ['forceplate', 'passiveOpto'], # otp_dir
                                ['', '_preOpto_levels_incline'], # appdx
                                ['level', 'trialType'], # indep_var
                                ['forceplate_levels','preOpto_levels_incline'], # appdx2
                                [FigConfig.colour_config['headbars'], FigConfig.colour_config['homolateral'][2]], 
                                ['dashed', 'solid'],
                                ['Force sensors', 'Treadmill'],
                                [Config.forceplate_config, Config.passiveOpto_config],
                                [B_FP, B_TRDM]
                                )):
    df = pd.read_csv(os.path.join(Config.paths[f"{otp_dir}_output_folder"], yyyymmdd + f'_limbPositionRegressionArray{appdx}.csv'))
    modLIN = pd.read_csv(os.path.join(Config.paths[f"{otp_dir}_output_folder"], yyyymmdd + f'_limbPositionRegressionArray_MIXEDMODEL_linear_{appdx2}.csv'), index_col=0)

    df[indep_var] = [-int(x[3:]) for x in df[indep_var]]
    zero_incline_centred = - np.nanmean(df[indep_var])
    model_Yzero = modLIN.loc[f'Intercept_{dep_var}','Estimate'] + np.nanmean(df[dep_var]) + modLIN.loc[f'param_centred_{dep_var}','Estimate'] * zero_incline_centred

    # APPROXIMATE WITH A FUNCTION
    from scipy.optimize import curve_fit
    from scipy.stats import t
    def linear(x,A,B):
        return A - B*x
    
    if otp_dir == 'forceplate':
        df[dep_var] = (df[dep_var] - model_Yzero)/cfg["px_per_cm"]
    else:
        df[dep_var] = (df[dep_var] - model_Yzero)/cfg["px_per_cm"][dep_var[:-1]] # convert to cm

    df['indep_bins'], bin_edges = pd.cut(df[indep_var], bins=bin_edges, retbins = True, labels = False)
    
    summary = df.groupby('indep_bins')[dep_var].agg(['std', 'sem']).reset_index()
    summary['bin_x'] = [np.mean((bin_edges[i],bin_edges[i+1])) for i in range(bin_edges.shape[0]-1)]
    
    x_pred = np.linspace(summary['bin_x'].min(), summary['bin_x'].max(), endpoint=True)
    
    # PER-MOUSE
    for m in df['mouseID'].unique():
        df_sub = df.loc[df['mouseID']==m, :].copy()
        mask_sub = ~np.isnan(df_sub[dep_var].values)
        popt_sub,_ = curve_fit(linear, 
                              df_sub[indep_var].values[mask_sub], 
                              df_sub[dep_var].values[mask_sub], 
                              p0=(np.nanmean(df_sub[dep_var].values),
                        (np.nanmax(df_sub[dep_var].values)-np.nanmin(df_sub[dep_var].values))/(np.nanmax(df_sub[indep_var].values)-np.nanmin(df_sub[indep_var].values)),
                        ))
        _, B_fit_sub = popt_sub
        B_list.append(B_fit_sub)
  
    
    mask = ~np.isnan(df[dep_var].values)
    popt,pcov = curve_fit(linear, df[indep_var].values[mask], df[dep_var].values[mask], p0=(np.nanmean(df[dep_var].values),
                                                                                  (np.nanmax(df[dep_var].values)-np.nanmin(df[dep_var].values))/(np.nanmax(df[indep_var].values)-np.nanmin(df[indep_var].values)),
                                                                                  ))
    A_fit, B_fit = popt
    print(f"Linear fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}")
    y_pred = linear(x_pred, *popt)
    
    # 95% confidence intervals
    ids = np.linspace(0, len(x_pred)-1, group_num, dtype=int)
    summary['ci95_hi'] = y_pred[ids] + summary['std'] + 1.96*summary['sem']
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
    
    ax.set_xlim(-50,50)
    ax.set_xticks([-40,-20,0,20,40], labels = [-40,-20,0,20,40])
    ax.set_ylim(-2,2)
    
    ## plot horizontal line as a legend
    ax.hlines(1.9, 6 - (i*50), 37 - (i*46), color = clr, ls = lnst, lw = 0.7)
    
    # COMBINE DATA
    x_data_comb = np.concatenate((x_data_comb, df[indep_var].values[mask]))
    y_data_comb = np.concatenate((y_data_comb, df[dep_var].values[mask]))
    n_data.append(df[indep_var].values[mask].shape[0])
    ssrs.append(np.sum((df[dep_var].values[mask] - linear(df[indep_var].values[mask], *popt))**2))
    
    # COMPUTE INDIVIDUAL P-VALUES
    std_err = np.sqrt(np.diag(pcov)) # standard deviations in 
    t_values = popt/std_err
    dof = max(0, len(df[dep_var].values)-len(popt))   
    p_values = [2 * (1 - t.cdf(np.abs(t_val), dof)) for t_val in t_values]
    print(f"{otp_dir} p-values: A_p = {p_values[0]:.3e}, B_p = {p_values[1]:.3e}")
    
plt.tight_layout()

# COMPUTE WILCOXON RANK SUM TEST STATISTIC
from scipy.stats import mannwhitneyu

stat, p_value = mannwhitneyu(B_FP, B_TRDM)
print(stat, p_value)


ax.text(-45,2, "treadmill ", ha = 'left', 
        color = FigConfig.colour_config['homolateral'][2], 
        fontsize = 5)
p_text = "vs sensors: " + ('*' * (p_value < FigConfig.p_thresholds).sum())
if (p_value < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
ax.text(-6,2, p_text, ha = 'left', 
        color = FigConfig.colour_config['headbars'], 
        fontsize = 5)


ax.text(-50.5,2.3,'anterior', ha = 'right', fontsize = 5)
ax.text(-50.5,-2.5,'posterior', ha = 'right', fontsize = 5)
ax.set_title("Surface slope trials")

ax.set_xlabel('Surface slope (deg)')
ax.set_ylabel('Horizontal position of\nRH foot (cm)')
plt.tight_layout()
fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"forceplate_foorPosition_vs_slope_COMPARISON.svg",
            dpi=300,
            transparent=True)