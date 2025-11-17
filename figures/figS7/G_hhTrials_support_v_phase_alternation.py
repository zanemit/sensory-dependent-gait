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
from scipy.optimize import curve_fit
from processing import data_loader, utils_math, utils_processing, treadmill_linearGLM
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

# from sklearn.model_selection import train_test_split
# import statsmodels.api as sm
from scipy.stats import wilcoxon

# outcome_variable = 'limbSupportPC3' # should change
ref = 'lH1'

yyyymmdd = '2022-08-18'
rH_category = ['alt']#["Llead", "Rlead"] #"sync" #"alt"
rH_categ_str = 'alt'#'asym'
appdx = f"_rH{rH_categ_str}"

ytexts = ('3-limb', 'diagonal')
yticks = [-0.3, 0, 0.3] 
ylims=(-0.3, 0.3)
xlims = (0, 2*np.pi)

datafull = data_loader.load_processed_data(dataToLoad = 'strideParamsMerged', 
                                                yyyymmdd = yyyymmdd, 
                                                limb = ref, 
                                                appdx = appdx)[0]
dependent_col = 'limbSupportPC1'
limb = 'lF0'

# convert to radians
datafull[limb] = datafull[limb]*2*np.pi
datafull[limb] = np.where(datafull[limb]<0, datafull[limb]+(2*np.pi), datafull[limb]) 

# subset category
datafull = datafull[datafull['rH0_categorical'].isin(rH_category)].copy().reset_index(drop=True)

# compute phase components
datafull[f'{limb}_sin'] = np.sin(datafull[f'{limb}'])
datafull[f'{limb}_cos'] = np.cos(datafull[f'{limb}'])
df_sub = datafull[[f'{limb}_sin', f'{limb}_cos', f'{limb}', dependent_col, 'mouseID']].copy()
df_sub = df_sub.dropna()
print("Total number of rows:", df_sub.shape[0])

# define a linear model with two predictors
def linear_model(X, a, b, c):
    x1, x2 = X
    return a + b*x1 + c*x2

mask_threshold = 300
num_bins = min([int(round(df_sub.shape[0]/mask_threshold, 0)), 20])

bins= np.linspace(0, 2*np.pi, num_bins+1)
bin_centres = (bins[:-1]+bins[1:])/2
binned_means = np.zeros((len(df_sub['mouseID'].unique()), num_bins))
stride_nums = np.zeros(num_bins)
cos_mice = []; sin_mice = []
for i, m in enumerate(df_sub['mouseID'].unique()):
    df_m = df_sub[df_sub['mouseID'] == m].copy()
    indep_var = df_m[f'{limb}'].values
    dep_var = df_m[dependent_col].values
    
    # quantify variability
    popt, pcov = curve_fit(linear_model, (df_m[f'{limb}_sin'].values, df_m[f'{limb}_cos'].values), dep_var)

    INT_fit, SIN_fit, COS_fit = popt
    cos_mice.append(COS_fit)
    sin_mice.append(SIN_fit)
    
    for j in range(num_bins):
        mask = (indep_var >= bins[j]) & (indep_var <bins[j+1])
        binned_means[i,j] = np.nanmean(dep_var[mask])
        stride_nums[j] += dep_var[mask].shape[0]
        
total_mask = stride_nums>=mask_threshold
mean_vals = np.nanmean(binned_means, axis=0)[total_mask]
sem_vals = scipy.stats.sem(binned_means, axis=0, nan_policy='omit')[total_mask]
ci95 = sem_vals*1.96

fig,ax = plt.subplots(1,1,figsize=(1.45,1.4))
clr=FigConfig.colour_config['homolateral'][2]
ax.fill_between(bin_centres[total_mask], mean_vals-ci95, mean_vals+ci95, facecolor=clr, alpha=0.2, edgecolor=None)
ax.plot(bin_centres[total_mask],
        mean_vals,
        color=clr,
        lw=1.5,
        label='Mean')

ax.set_xlim(bins.min(), bins.max())


cos_wilcoxon = wilcoxon(cos_mice)
sin_wilcoxon = wilcoxon(sin_mice)
print(f"Wilcoxon COS t={cos_wilcoxon[0]}, p={cos_wilcoxon[1]}, SIN t={cos_wilcoxon[0]}, p={sin_wilcoxon[1]}")

for i, (coord, coord_str) in enumerate(zip([sin_wilcoxon, cos_wilcoxon], ['sin', 'cos'])):
    ptext = '*' * (coord[1]<np.asarray(FigConfig.p_thresholds)).sum() if (coord[1]<np.asarray(FigConfig.p_thresholds)).sum()!=0 else 'n.s.'
    stat_text = f"phase {coord_str}: {ptext}" if i==0 else f"{coord_str}: {ptext}"
    ax.text(xlims[0] + (0.01 * (xlims[1]-xlims[0]))+1.4*np.pi*i,
            ylims[1] - (0.05* (ylims[1]-ylims[0])),
            stat_text,
            fontsize=5)
    
ax.set_ylabel(dependent_col[-3:])
ax.set_ylim(ylims[0], ylims[1])
ax.set_yticks(yticks)
ax.set_xlabel("Left homolateral\nphase (rad)")
ax.set_xticks([0,0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi], 
              labels=['0', '0.5π', 'π', '1.5π', '2π'])

# add y labels
ax.text(-0.2*np.pi,
        ylims[0]-(ylims[1]-ylims[0])*0.16,
        ytexts[0],
        fontsize=5,
        ha='right', fontweight='bold',
        color=FigConfig.colour_config['homolateral'][1])
ax.text(-0.2*np.pi,
        ylims[1]+(ylims[1]-ylims[0])*0.09,
        ytexts[1],
        fontsize=5,
        ha='right',fontweight='bold',
        color=FigConfig.colour_config['homologous'][1])

ax.set_title('L-R alternation', fontsize=6)
plt.tight_layout()

figtitle = f"passiveOpto_limbSupportPC{dependent_col[-1]}_vs_homolateral_{rH_categ_str}.svg"
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], figtitle), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)        
    