import sys
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_math, utils_processing
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

yyyymmdd = ['2021-10-23', '2022-05-06'] 
appdx = "" 
refLimb = "lH1"
mouselist = None
kdeplot = False
outputDir = Config.paths["mtTreadmill_output_folder"]
kde_bin_num = 100 
    
if refLimb == 'lH1':
    limblist = {'homologous': 'rH0', 'homolateral': 'lF0', 'diagonal': 'rF0'}
    side_str = 'right'
elif refLimb == 'rH1':
    limblist = {'homologous': 'lH0', 'homolateral': 'rF0', 'diagonal': 'lF0'}
    side_str = 'left'

if type(yyyymmdd) != list:
    yyyymmdd = [list]
for i, yyyymmdd_i in enumerate(yyyymmdd):
    if i == 0:
        df, _, _ = data_loader.load_processed_data(outputDir, 
                                        dataToLoad = 'strideParams', 
                                        appdx = appdx,
                                        yyyymmdd = yyyymmdd_i,
                                        limb = refLimb)
    else:
        df2, _, _ = data_loader.load_processed_data(outputDir, 
                                        dataToLoad = 'strideParams', 
                                        appdx = appdx,
                                        yyyymmdd = yyyymmdd_i,
                                        limb = refLimb)
        df = pd.concat([df, df2], ignore_index = True)

if type(mouselist) == list: # list provided
    for m in np.setdiff1d(np.unique(df['mouseID']), mouselist):
        df = df[df['mouseID'] != m]
        
df['hind_phase'] = df[limblist['homologous']]
df['fore_phase'] = utils_processing.compute_forelimb_phase(ref_phases = df[limblist['homolateral']], nonref_phases = df[limblist['diagonal']])
limbs = ['hind', 'fore']
limb_str = ['homologous', 'diagonal']

mice = np.unique(df['mouseID'])

fracKDE_R_after_L_across_mice = np.empty((mice.shape[0], len(limbs)))
fracKDE_R_after_L_across_mice[:] = np.nan
frac_R_after_L_across_mice = np.empty((mice.shape[0], len(limbs)))
frac_R_after_L_across_mice[:] = np.nan

# limb phase restrictions! = where it makes sense to look at lateralisation
upper_lim = 0.375 
lower_lim = 0.05

fig, ax = plt.subplots(1,1, figsize = (1.5,1.4))

# INDIVIDUAL PLOTS
rows, cols = utils_processing.get_num_subplots(mice.shape[0])

for il, limb in enumerate(limbs):
    phase_str = f'{limb}_phase'
    df_sub = df.iloc[np.where((df[phase_str]<=upper_lim) & (df[phase_str]>=-upper_lim))[0], :]
    
    df_sub = df_sub.iloc[np.where((df_sub[phase_str]>=lower_lim) | (df_sub[phase_str]<=-lower_lim))[0], :]
    print(f"{limb} dataframe entries: {df_sub.shape[0]}, {100*df_sub.shape[0]/df.shape[0]:.2f}% of all steps")
    
    # INDIVIDUAL PLOTS
    i = 0; j = 0
    
    for im, m in enumerate(mice):
        df_sub_m = df_sub[df_sub['mouseID'] == m]
        values = df_sub_m[phase_str]
        kde_bins, kde = utils_math.von_mises_kde_exact(values*2*np.pi, 10, kde_bin_num)
        kde_bins_points = np.concatenate((kde_bins, [kde_bins[0]]))
        kde_points = np.concatenate((kde, [kde[0]]))
        
        i = im // cols
        j = im % cols
        
        # right after left
        kde_points_suprazero = kde_points[:-1][kde_bins_points[:-1]>0]
        kde_bins_points_suprazero = kde_bins_points[:-1][kde_bins_points[:-1]>0]
        area_suprazero = np.trapz(kde_points_suprazero, kde_bins_points_suprazero)
        
        # left after right
        kde_points_subzero = kde_points[:-1][kde_bins_points[:-1]<0]
        kde_bins_points_subzero = kde_bins_points[:-1][kde_bins_points[:-1]<0]
        area_subzero = np.abs(np.trapz(kde_points_subzero, kde_bins_points_subzero))
        
        area_total = area_suprazero + area_subzero
        
        fracKDE_R_after_L_across_mice[im, il] = area_suprazero/area_total
        
        # compute the same but as regular fractions!
        # right after left
        points_suprazero = (values>0).sum()
        points_subzero = (values<0).sum()
        frac_R_after_L_across_mice[im, il] = points_suprazero / (points_suprazero + points_subzero)
    
    if kdeplot:
        frac_arr = fracKDE_R_after_L_across_mice
    else:
        frac_arr = frac_R_after_L_across_mice
    
    clr = FigConfig.colour_config[limb_str[il]][2]
    ax.boxplot(frac_arr[:, il][~np.isnan(frac_arr[:, il])], positions = [il], 
               medianprops = dict(color = clr, linewidth = 1, alpha = 0.8),
               boxprops = dict(color = clr, linewidth = 1, alpha = 0.8), 
               capprops = dict(color = clr, linewidth = 1, alpha = 0.8),
               whiskerprops = dict(color = clr, linewidth = 1, alpha = 0.8), 
               flierprops = dict(mec = clr, linewidth = 1, alpha = 0.8, ms=2))
    ax.scatter(np.repeat(il, 
               frac_arr.shape[0]), 
               frac_arr[:, il], 
               color =  clr, alpha = 0.8, s = 5, zorder = 3)

for i in range(frac_arr.shape[0]):
    clr = [FigConfig.colour_config[limb_str[0]][2], FigConfig.colour_config[limb_str[1]][2]]
    ax.plot([0,1], frac_arr[i, :], linewidth = 1, color = '#b17242', alpha = 0.4)
ax.axhline(0.5, 0.2, 0.8, ls = 'dashed', color = 'grey')
ax.set_title("Motorised treadmill")

ax.set_ylim(0,1)
ax.set_yticks(np.linspace(0,1, 5, endpoint=True))
ax.set_xticklabels(limbs)
ax.set_ylabel(f"Fraction of\n{side_str}-leading strides")
fig.tight_layout()
fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_lateralisation_fractions_{refLimb}_kde{kdeplot}{appdx}.svg", 
            dpi = 300, 
            bbox_inches = 'tight')  
