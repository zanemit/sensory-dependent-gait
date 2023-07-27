import sys
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_math, utils_processing
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

mouselist1 = ['BAA1098955','BAA1099004','FAA1034468','FAA1034469','FAA1034471',
              'FAA1034472','FAA1034570','FAA1034572','FAA1034573','FAA1034575',
              'FAA1034576','FAA1034665'] 
yyyymmdd1 = '2022-02-26'
mouselist2 = ['FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 
              'FAA1034868', 'FAA1034869', 'FAA1034942', 'FAA1034944', 'FAA1034945', 
              'FAA1034947', 'FAA1034949'] #2022-08-18
yyyymmdd2 = '2022-08-18'

outputDir = Config.paths["passiveOpto_output_folder"]
mouselist3 = [m for mouselist in [mouselist1, mouselist2] for m in mouselist]
yyyymmdd3 = [yyyymmdd1, yyyymmdd2]
appdx = "" 
param = 'homolateral_gait'

# RUNNING PARAMS
yyyymmdd = yyyymmdd3
refLimb = 'lH1'
mouselist = mouselist3 
appdx = appdx
kdeplot = False
kde_bin_num = 100
injectionsideplot = True #THIS DETERMINES WHETHER INJ SIDE GETS CIRCLED
    
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

df = df[df['speed'] > np.nanpercentile(df['speed'],50) ]      
df['hind_phase'] = df[limblist['homologous']]
df['fore_phase'] = utils_processing.compute_forelimb_phase(ref_phases = df[limblist['homolateral']], nonref_phases = df[limblist['diagonal']])
limbs = ['hind', 'fore']
limb_str = ['homologous', 'diagonal']

mice = np.unique(df['mouseID'])
mice_R_inj = np.asarray([1 if m in Config.injection_config['right_inj_imp'] else 0 for m in mice]).astype(bool)
mice_both_inj = np.asarray([1 if m in Config.injection_config['both_inj_left_imp'] else 0 for m in mice]).astype(bool)

fracKDE_R_after_L_across_mice = np.empty((mice.shape[0], len(limbs)))
fracKDE_R_after_L_across_mice[:] = np.nan
frac_R_after_L_across_mice = np.empty((mice.shape[0], len(limbs)))
frac_R_after_L_across_mice[:] = np.nan

# limb phase restrictions! = where it makes sense to look at lateralisation
upper_lim = 0.375 
lower_lim = 0.05

fig, ax = plt.subplots(1,1, figsize = (1.5,1.4))

for il, limb in enumerate(limbs):
    phase_str = f'{limb}_phase'
    df_sub = df.iloc[np.where((df[phase_str]<=upper_lim) & (df[phase_str]>=-upper_lim))[0], :]
    
    df_sub = df_sub.iloc[np.where((df_sub[phase_str]>=lower_lim) | (df_sub[phase_str]<=-lower_lim))[0], :]
      
    for im, m in enumerate(mice):
        df_sub_m = df_sub[df_sub['mouseID'] == m]
        values = df_sub_m[phase_str]
        kde_bins, kde = utils_math.von_mises_kde_exact(values*2*np.pi, 10, kde_bin_num)
        kde_bins_points = np.concatenate((kde_bins, [kde_bins[0]]))
        kde_points = np.concatenate((kde, [kde[0]]))
        
        
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
    
    print((frac_arr[:, il]>0.5).sum()/frac_arr[:, il].shape[0])
    
    clr = FigConfig.colour_config[limb_str[il]][2]
    ax.boxplot(frac_arr[:, il], positions = [il], 
               medianprops = dict(color = clr, linewidth = 1, alpha = 0.8),
               boxprops = dict(color = clr, linewidth = 1, alpha = 0.8), 
               capprops = dict(color = clr, linewidth = 1, alpha = 0.8),
               whiskerprops = dict(color = clr, linewidth = 1, alpha = 0.8), 
               flierprops = dict(mec = clr, linewidth = 1, alpha = 0.8, ms=2))
    ax.scatter(np.repeat(il, 
               frac_arr.shape[0]), 
               frac_arr[:, il], 
               color =  clr, alpha = 0.8, s = 5, zorder = 3)
    if injectionsideplot:
        ax.scatter(np.repeat(il, 
                   frac_arr[mice_R_inj].shape[0]), 
                   frac_arr[mice_R_inj, il], 
                   facecolors =  'teal', 
                   edgecolors = 'none',
                   alpha = 0.8, s = 6, zorder = 4,
                   linewidth = 0.5, label = 'right')
        ax.scatter(np.repeat(il, 
                   frac_arr[mice_both_inj].shape[0]), 
                   frac_arr[mice_both_inj, il], 
                   facecolors =  'black', 
                   edgecolors = 'none',
                   alpha = 0.8, s = 6, zorder = 3,
                   linewidth = 0.5, label = 'both')
        if il == 0:
            fig.legend(bbox_to_anchor=(0.35,-0.13,0.6,0.2),
                       mode="expand", 
                       title = "injection side", 
                       ncol = 2,
                       handletextpad = 0.1
                       )
    
for i in range(frac_arr.shape[0]):
    clr = [FigConfig.colour_config[limb_str[0]][2], FigConfig.colour_config[limb_str[1]][2]]
    ax.plot([0,1], frac_arr[i, :], linewidth = 1, color = '#b17242', alpha = 0.4)
ax.axhline(0.5, 0.2, 0.8, ls = 'dashed', color = 'grey')

ax.set_ylim(0,1)
ax.set_yticks(np.linspace(0,1, 5, endpoint=True))
ax.set_xticklabels(['pelvic', 'shoulder'])
ax.set_ylabel(f"Fraction of\n{side_str}-leading steps")
ax.set_title("Passive treadmill")
fig.tight_layout()

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_lateralisation_fractions_{refLimb}_{injectionsideplot}_kde{kdeplot}{appdx}.svg", 
            dpi = 300, 
            bbox_inches = 'tight')  

