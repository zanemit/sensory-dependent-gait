#%% TRYING TO MODEL LIMB-SUPPORT-PCs AS A FUNCTION OF HMLT PHASE
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os 
from matplotlib import pyplot as plt
import scipy.stats

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

import scipy.stats
from processing import data_loader, utils_math, utils_processing, treadmill_linearGLM
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

yyyymmdd = '2022-08-18'
ref = 'lH1'
appdx = "_incline_COMBINEDtrialType"

datafull = data_loader.load_processed_data(dataToLoad = 'strideParamsMerged', 
                                                yyyymmdd = yyyymmdd, 
                                                limb = ref, 
                                                appdx = appdx)[0]
    
dependent_col = 'limbSupportPC3'
limb = 'lF0'

if 'PC3' in dependent_col:
    ytlt = "Diagonal support\n(PC3)"
elif "PC4" in dependent_col:
    ytlt = "Single-leg support\n(PC4)"
else:
    ytlt = "DEFINE LABEL!!!"

# convert to radians
datafull[limb] = datafull[limb]*2*np.pi
datafull[limb] = np.where(datafull[limb]<0, datafull[limb]+(2*np.pi), datafull[limb]) 

# compute phase components
datafull[f'{limb}_sin'] = np.sin(datafull[f'{limb}'])
datafull[f'{limb}_cos'] = np.cos(datafull[f'{limb}'])
df_sub = datafull[[f'{limb}_sin', f'{limb}_cos', f'{limb}', dependent_col, 'trialType']]
df_sub = df_sub.dropna()

y = df_sub[dependent_col]

if '3' in dependent_col or '6' in dependent_col:
    yticks = [-0.1, 0, 0.1] 
    ylims=(-0.1, 0.12)
elif '4' in dependent_col or '1' in dependent_col:
    yticks = [-0.05, 0, 0.05, 0.10, 0.15] 
    ylims=(-0.05, 0.15)
else:
    yticks = [-0.1, 0, 0.1] 
    ylims=(-0.1, 0.1)
    # raise ValueError("ylim not specified!")
xlims = (0, 2*np.pi)
    
# ########################## PLOT ##########################
num_bins = 15
fig,ax = plt.subplots(1,1,figsize=(1.45,1.35))

for i, (trialtype, lnst, clr, lbl) in enumerate(zip(['slope', 'headHeight'],
                           ['solid', 'dashed'],
                           [FigConfig.colour_config['homolateral'][2], FigConfig.colour_config['greys'][0]],
                           ['slope', 'head h.']
                           )):
    mask = df_sub['trialType']==trialtype
    y_trial = y[mask].copy().reset_index(drop=True)
    indep_var = df_sub[f'{limb}'][mask].copy().reset_index(drop=True)
    bins = np.linspace(min(indep_var), max(indep_var), num_bins+1)
    bin_centres = (bins[:-1]+ bins[1:])/2
    
    print(np.array([len(y_trial[(indep_var>=bins[i]) & (indep_var<bins[i+1])]) for i in range(num_bins)]))
    binned_ymeans = np.array([y_trial[(indep_var>=bins[i]) & (indep_var<bins[i+1])].mean() for i in range(num_bins)])
    ci = np.array([1.96*scipy.stats.sem(y_trial[(indep_var>=bins[i]) & (indep_var<bins[i+1])]) for i in range(num_bins)])
    
    ax.fill_between(bin_centres, binned_ymeans-ci, binned_ymeans+ci, 
                    facecolor=clr, edgecolor=None, alpha=0.2)
    ax.plot(bin_centres, binned_ymeans,  lw=1.5, color=clr, ls=lnst)
    
    if '3' in dependent_col:
        print(f"Max bin:{bin_centres[np.argmax(binned_ymeans)]/np.pi}")
    elif '4' in dependent_col:
        print(f"Min bin:{bin_centres[np.argmin(binned_ymeans)]/np.pi}")
    
    # ADD TEXT
    ax.hlines(ylims[1]-0.18*(ylims[1]-ylims[0]), 
              xlims[0] + (0.58 * (xlims[1]-xlims[0])) - 1.12*np.pi*i, 
              xlims[0] + (0.77 * (xlims[1]-xlims[0])) - 0.92*np.pi*i, 
              color = clr, 
              ls = lnst, 
              lw = 0.5)
    ax.text(xlims[0] + (0.58 * (xlims[1]-xlims[0])) - 1.12*np.pi*i,
            ylims[1] - (0.16* (ylims[1]-ylims[0])),
            lbl,
            color=clr,
            fontsize=5)
 
# ########################## ADD STATS ########################## 

stats_path = os.path.join(Config.paths['passiveOpto_output_folder'], 
   f"{yyyymmdd}_contCoefficients_mixedEffectsModel_linear_{dependent_col}_vs_{limb}_trialType_randSlopesCosSin.csv")

stats = pd.read_csv(stats_path, index_col=0)

ptext = '*' * (stats.loc['trialTypeslope', 'Pr(>|t|)']<np.asarray(FigConfig.p_thresholds)).sum()
if (stats.loc['trialTypeslope', 'Pr(>|t|)']<np.asarray(FigConfig.p_thresholds)).sum() == 0:
    ptext = 'n.s.'
ax.text(xlims[0] + (0.43 * (xlims[1]-xlims[0])),
        ylims[1] - (0.255* (ylims[1]-ylims[0])),
        f"vs          trials:\n{ptext}",
        fontsize=5)

for i, coord in enumerate(['sin', 'cos']):
    ptext = '*' * (stats.loc[f'limb_{coord}', 'Pr(>|t|)']<np.asarray(FigConfig.p_thresholds)).sum()
    if (stats.loc[f'limb_{coord}', 'Pr(>|t|)']<np.asarray(FigConfig.p_thresholds)).sum() == 0:
        ptext = 'n.s.'
    ax.text(xlims[0] + (0.01 * (xlims[1]-xlims[0]))+1.1*np.pi*i,
            ylims[1] - (0.05* (ylims[1]-ylims[0])),
            f"LF {coord}: {ptext}",
            fontsize=5)

ax.set_ylabel(ytlt)
ax.set_ylim(ylims[0], ylims[1])
ax.set_yticks(yticks)
ax.set_xlabel("Relative LF phase\n(rad)")
ax.set_xticks([0,0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi], 
              labels=['0', '0.5π', 'π', '1.5π', '2π'])
plt.tight_layout()
    
figtitle = f"passiveOpto_limbSupportPC{dependent_col[-1]}_vs_homolateral.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
