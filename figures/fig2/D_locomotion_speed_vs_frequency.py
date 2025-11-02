import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
# import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

yyyymmdd = '2022-08-18'
df, yyyymmdd = data_loader.load_processed_data(dataToLoad = 'locomParamsAcrossMice', 
                                               yyyymmdd = yyyymmdd,
                                               appdx = '')

# df = df[df['headLVL']=='rl6']
# trialType = 'headHeight'
clr = FigConfig.colour_config['greys'][0]
lnst = 'solid'
freqs = [10,20,30,40,50]

# df = df[df['trialType']==trialType]
for m in np.setdiff1d(np.unique(df['mouseID']), Config.passiveOpto_config['mice']):
    df = df[df['mouseID'] != m]

df['stimFreq_num'] = [int(x[:-2]) for x in df['stimFreq']]

subplot_num = 2
fig, ax = plt.subplots(1,2, figsize = (1*2, 1.25))

mice = np.unique(df['mouseID'])
arr = np.empty((mice.shape[0], len(freqs), 2)) # mouse_num, group_num, mean/max
arr[:] = np.nan

for im, m in enumerate(mice):
    df_sub = df[df['mouseID'] == m]
    for xf, f in enumerate(freqs):
        df_sub2 = df_sub[df_sub['stimFreq_num'] == f]
        
        arr[im, xf, 0] = np.nanmean(df_sub2['meanSpeed'])
        arr[im, xf, 1] = np.nanmean(df_sub2['maxSpeed'])

for i, (colname, label, ylims, tlt) in enumerate(zip(['meanSpeed', 'maxSpeed'], 
                                         ['mean', 'max'],
                                         [(0,80), (80,160)],
                                         ['Mean speed', 'Max speed']
                                         )):
    
    meanval = np.nanmean(arr[:,:,i], axis = 0)
    sem = scipy.stats.sem(arr[:,:,i], axis = 0, nan_policy = 'omit')  
    ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., arr[:,:,i].shape[0]-1) 
    ax[i].fill_between(freqs, 
                    meanval - ci,
                    meanval + ci, 
                    facecolor = clr, 
                    alpha = 0.2)
    
    ax[i].plot(freqs,
            np.nanmean(arr[:,:,i], axis = 0),
            color = clr,  
            linewidth = 1.5,
            label = label,
            )

    ax[i].set_title(tlt)
    ax[i].set_xlabel('Stimulation\nfrequency (Hz)')
    ax[i].set_xlim(5, 50)
    ax[i].set_xticks([10,30,50])
    ax[i].set_ylim(ylims[0], ylims[1])
    ax[i].set_yticks(np.linspace(ylims[0], ylims[1], 3, endpoint=True))
    try:
        ax[i].get_legend().remove()  
    except:
        pass
    
    # p value
    slope_enforced = 'slopeENFORCED'
    rand_effect = 'Slope'
    mod = pd.read_csv(os.path.join(Config.paths['passiveOpto_output_folder'], f"{yyyymmdd}_mixedEffectsModel_linear_{colname}_stimFreq_{slope_enforced}_rand{rand_effect}.csv"), index_col=0)
    t = mod.loc['indep_var_centred', 't value']
    p = mod.loc['indep_var_centred', 'Pr(>|t|)']
    print(f"{colname}: mean={mod.loc['indep_var_centred', 'Estimate']:.4g}, SEM={mod.loc['indep_var_centred', 'Std. Error']:.4g}, t({mod.loc['indep_var_centred', 'df']:.0f})={t:.3f}, p={p:.3g}")
    x =0
    p_text = "n.s." if (p < FigConfig.p_thresholds).sum() == 0 else ('*' * (p < FigConfig.p_thresholds).sum())
    ax[i].text(0.5,
            1, 
            p_text, 
            ha = 'center', 
            color = clr, 
            transform=ax[i].transAxes,
            fontsize = 5)   


ax[0].set_ylabel("Speed (cm/s)") #label)
            
plt.tight_layout()

# lgd = fig.legend([([FigConfig.colour_config['homolateral'][0]],"solid"), 
#                 ([FigConfig.colour_config['homolateral'][0]],"dotted")], 
#                 ["median", "max"], 
#                 handler_map={tuple: AnyObjectHandler()}, 
#                 loc = 'upper center', bbox_to_anchor=(0.3,0.9,0.7,0.2),
#                 mode="expand", borderaxespad=0, ncol = 2)

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"forceplate_locomParamsAcrossMice_speed_vs_stimFreq_rl6.svg",
            dpi = 300,
            transparent = True)


 
