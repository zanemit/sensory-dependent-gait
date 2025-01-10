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
df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                               dataToLoad="forceplateAngleParams", 
                                               yyyymmdd = yyyymmdd, 
                                               appdx = '_headHW')
modQDRhw = pd.read_csv(Path(Config.paths["forceplate_output_folder"])/f"{yyyymmdd}_mixedEffectsModel_quadratic_snoutBodyAngle_headHW.csv")

mice = np.unique(df['mouseID'])

fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.47))
for m in mice:
    df_sub = df[df['mouseID'] == m]
    headHWs = np.unique(df_sub['headHW'])
    bodyAngles = [np.nanmean(df_sub[df_sub['headHW']==h]['snoutBodyAngle']) for h in np.unique(df_sub['headHW'])]
    ax.plot(headHWs, 
            bodyAngles, 
            color = FigConfig.colour_config['headbars'], 
            alpha=0.4, 
            linewidth = 0.7)

# APPROXIMATE WITH A FUNCTION
from scipy.optimize import curve_fit
from scipy.stats import t
def exp_decay(x,A,B,k):
    return A - B * np.exp(-k*x)
def linear_fit(x,A,B):
    return A + B * x

x_pred = np.linspace(np.nanmin(df['headHW'].values), np.nanmax(df['headHW'].values), endpoint=True)

mask = ~np.isnan(df['snoutBodyAngle'].values)
popt,pcov = curve_fit(exp_decay, df['headHW'].values[mask], df['snoutBodyAngle'].values[mask], p0=(np.nanmax(df['snoutBodyAngle'].values),
                                                                              np.nanmax(df['snoutBodyAngle'].values)-np.nanmin(df['snoutBodyAngle'].values),
                                                                              1/np.nanmean(df['headHW'].values)))
A_fit, B_fit, k_fit = popt
print(f"Exp decay fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}, k = {k_fit:.3f}")
ax.plot(x_pred, 
              exp_decay(x_pred, *popt), 
              linewidth=1.5, 
              color=FigConfig.colour_config['headbars'])
# print(f"SHIFT: {exp_decay(x_pred, *popt)[-1]-exp_decay(x_pred, *popt)[0]}")
std_err = np.sqrt(np.diag(pcov)) # standard deviations in 
t_values = popt/std_err
dof = max(0, len(df['snoutBodyAngle'].values)-len(popt))   
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
                 179-(i_p*4), 
                 f"{exp_d_param}: {p_text}", 
                 ha = 'center', 
                 color = FigConfig.colour_config['headbars'],
                 fontsize = 5)

ax.set_xlim(0,1.2)
ax.set_ylim(140,180)
ax.set_ylabel('Snout-hump angle\n(deg)')
ax.set_xlabel('Weight-adjusted\nhead height')
ax.set_xticks([0,0.4,0.8,1.2])
ax.set_yticks([140,150,160,170,180])
# ax.text(0.6,179, p_text, ha = 'center', color = FigConfig.colour_config['headbars'])
ax.set_title("Head height trials")

plt.tight_layout()
fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"forceplate_snoutBodyAngles_vs_headHW.svg",
            transparent = True,
            dpi=300)



