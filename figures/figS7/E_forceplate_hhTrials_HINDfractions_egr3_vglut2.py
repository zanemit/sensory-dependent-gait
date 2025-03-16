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

param = 'headHW'

fig, axes = plt.subplots(1, 1, figsize=(1.3, 1.47))

yyyymmdds = ['2023-11-06', '2021-10-26']
ylin_last = []

limb_clr = 'homolateral'
limb_str = 'hind_weight_frac'
tlt = 'Head height trials'
variable_str = 'hindWfrac'

mod_type = 'linear' if 'hind' in limb_str else 'quadratic'

for yyyymmdd, lnst in zip(
        yyyymmdds,
        ['solid', 'dashed']
                                    ):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="meanParamDF", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = f'_{param}')
    mice = np.unique(df['mouse'])
    
     
    mod = pd.read_csv(os.path.join(Config.paths['forceplate_output_folder'], f"{yyyymmdd}_mixedEffectsModel_{mod_type}_{variable_str}_{param}.csv"), index_col=0)
    meanclr = FigConfig.colour_config[limb_clr][2] if '2023' in yyyymmdd else 'grey'
    
    if '2023' in yyyymmdd:
        print(f"There are {mice.shape[0]} Egr3 mice")
        for im, m in enumerate(mice):
            df_sub = df[df['mouse'] == m]
            headHWs = np.unique(df_sub['param'])
            
            yvals = [np.nanmean(df_sub[df_sub['param']==h][limb_str]) for h in np.unique(df_sub['param'])]
            axes.plot(headHWs, 
                     yvals, 
                     color=meanclr,  
                     alpha=0.2, 
                     linewidth = 0.7)
            # print(f"{m}: {tlt} weight fraction changed by {(yvals[-1]-yvals[0]):.2f}")
        
        # p_text = '*' * (mod.loc[:,'Pr(>|t|)'].iloc[1] < FigConfig.p_thresholds).sum()
        # axes[i].text(0.65,1.5, f"head height: {p_text}", 
        #         ha = 'center', fontsize=5)
        print(mod.loc[:,'Pr(>|t|)'].iloc[1])

        axes.text(0.6,1.5,"MSA-def", 
                fontsize=5,color=meanclr)
        axes.hlines(1.5,0.6,1,color=meanclr, linestyle='solid', lw=1.5)
        axes.text(0.4, 1.5, "vs")
        
    else:
        axes.text(0.05,1.5,"CTRL", 
                fontsize=5,color=meanclr)
        axes.hlines(1.5,0.05,0.4,color=meanclr, linestyle='dashed', lw=1.5)
    
    # ADD MEANS
    
    x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)
    x_pred -= np.nanmean(df['param'])
    
    if mod_type == 'linear':
        y_pred = mod.loc['(Intercept)', 'Estimate'] + (mod.loc['param_centred', 'Estimate'] * x_pred) + np.nanmean(df[limb_str])
    else:
        y_pred = mod.loc['(Intercept)', 'Estimate'] + (mod.loc['poly(param_centred, 2, raw = TRUE)1', 'Estimate'] * x_pred) + (mod.loc['poly(param_centred, 2, raw = TRUE)2', 'Estimate'] * x_pred**2) + np.nanmean(df[limb_str])
   
    x_pred += np.nanmean(df['param'])
    axes.plot(x_pred, 
              y_pred, 
              linewidth=1.5, 
              color=meanclr,
              ls=lnst)
    print(f"{yyyymmdd}: On average, weight fraction changed by {(y_pred[-1]-y_pred[0]):.2f}")
    print(f"{yyyymmdd}: x change {(x_pred[-1]-x_pred[0]):.2f}, {len(x_pred)}")
 
############# ADD STATS ############# 
statspath = f"C:\\Users\\MurrayLab\\Documents\\Forceplate\\{yyyymmdds[0]}_x_{yyyymmdds[1]}_mixedEffectsModel_{mod_type}_{limb_str}_v_headHW_egr3_vglut2.csv"
stats = pd.read_csv(statspath, index_col =0)
# axes.text(0.4, 188, "vs")
ptext = "intercept: "
for j, (ptext, statscol) in enumerate(zip(
        ['intercept: ', 'slopes: '],
        ['trialtypevglut2', 'param_centred:trialtypevglut2'],
        )):
    p = stats.loc[statscol, "Pr(>|t|)"]
    print(p)
    if (p < np.asarray(FigConfig.p_thresholds)).sum() == 0:
        ptext += "n.s."    
    else:
        ptext += ('*' * (p < np.asarray(FigConfig.p_thresholds)).sum())
    axes.text(0.63,
             1.3-(j*0.15),
            ptext, ha='center',
            fontsize=5,
            color=FigConfig.colour_config[limb_clr][2])

############# ADD STATS ############# 

############# ADD STATS ############# 
# # APPROXIMATE WITH A FUNCTION
# from scipy.optimize import curve_fit
# from scipy.stats import t
# def exp_decay(x,A,B,k):
#     return A - B * np.exp(-k*x)
# def linear_fit(x,A,B):
#     return A + B * x

# x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)

# # if "hind" not in limb_str:
# popt,pcov = curve_fit(exp_decay, df['param'].values, df[limb_str].values, p0=(0.25,0.75,2))

# A_fit, B_fit, k_fit = popt
# print(f"Exp decay fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}, k = {k_fit:.3f}")
# # axes.plot(x_pred, 
# #               exp_decay(x_pred, *popt), 
# #               linewidth=2, 
# #               color=FigConfig.colour_config[limb_clr][2])
# # print(f"LAST VALUE: {exp_decay(x_pred, *popt)[-1]}")
# std_err = np.sqrt(np.diag(pcov)) # standard errors
# t_values = popt/std_err
# dof = max(0, len(df[limb_str].values)-len(popt))   
# p_values = [2 * (1 - t.cdf(np.abs(t_val), dof)) for t_val in t_values]
# print(f"p-values: A_p = {p_values[0]:.3e}, B_p = {p_values[1]:.3e}, k_p = {p_values[2]:.3e}")
# for i_p, (p, exp_d_param) in enumerate(zip(
#         p_values[1:], 
#         ["scale factor", "rate constant"],
#         )):
#     p_text = ('*' * (p < FigConfig.p_thresholds).sum())
#     if (p < FigConfig.p_thresholds).sum() == 0:
#         p_text += "n.s."
#     axes.text(0.63,
#                   1.3-(i_p*0.15), 
#                   f"{exp_d_param}: {p_text}", 
#                   ha = 'center', 
#                   color = FigConfig.colour_config[limb_clr][2],
#                   fontsize = 5)
    
# else:
#     popt,pcov = curve_fit(linear_fit, df['param'].values, df[limb_str].values, p0=(0.5,0))
#     A_fit, B_fit = popt
#     print(f"Linear fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}")
#     axes.plot(x_pred, 
#                   linear_fit(x_pred, *popt), 
#                   linewidth=2, 
#                   color=FigConfig.colour_config[limb_clr][2])
#     std_err = np.sqrt(np.diag(pcov)) # standard errors
#     t_values = popt/std_err
#     dof = max(0, len(df[limb_str].values)-len(popt))   
#     p_values = [2 * (1 - t.cdf(np.abs(t_val), dof)) for t_val in t_values]
#     print(f"p-values: A_p = {p_values[0]:.3e}, B_p = {p_values[1]:.3e}")
#     for i_p, (p, exp_d_param) in enumerate(zip(
#             p_values[1:], 
#             ["slope"],
#             )):
#         p_text = ('*' * (p < FigConfig.p_thresholds).sum())
#         if (p < FigConfig.p_thresholds).sum() == 0:
#             p_text += "n.s."
#         axes.text(0.6,
#                      1.46-(i_p*0.1), 
#                      f"{exp_d_param}: {p_text}", 
#                      ha = 'center', 
#                      color = FigConfig.colour_config[limb_clr][2],
#                      fontsize = 5)


############# ADD STATS ############# 

        
axes.set_xlabel('Weight-adjusted\nhead height')

axes.set_xticks([0,0.6,1.2])
axes.set_xlim(0,1.2)
axes.set_yticks([0,0.5,1.0,1.5])
axes.set_ylim(-0.1,1.5)
axes.set_title(tlt)

axes.set_ylabel("Hindlimb weight fraction")
# axes[2].set_ylabel("Total detected weight (%)")
plt.tight_layout()
    
fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'forceplateEGR3_foreHindHeadWeights_{param}_{limb_str}.svg'),
            transparent = True,
            dpi = 300)
