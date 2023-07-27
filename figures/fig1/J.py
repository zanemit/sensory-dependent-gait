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
                                               appdx = '_COMBINED')
# trialType = 'headHeight'
clr = FigConfig.colour_config['homolateral'][0]
lnst = 'solid'
freqs = [10,20,30,40,50]

# df = df[df['trialType']==trialType]
for m in np.setdiff1d(np.unique(df['mouseID']), Config.passiveOpto_config['mice']):
    df = df[df['mouseID'] != m]

df['stimFreq_num'] = [int(x[:-2]) for x in df['stimFreq']]

subplot_num = 2
fig, ax = plt.subplots(1,1, figsize = (1.55,1.5))

mice = np.unique(df['mouseID'])
arr = np.empty((mice.shape[0], len(freqs), 2)) # mouse_num, group_num, mean/max
arr[:] = np.nan

for im, m in enumerate(mice):
    df_sub = df[df['mouseID'] == m]
    for xf, f in enumerate(freqs):
        df_sub2 = df_sub[df_sub['stimFreq_num'] == f]
        
        arr[im, xf, 0] = np.nanmean(df_sub2['medianSpeed'])
        arr[im, xf, 1] = np.nanmean(df_sub2['maxSpeed'])
        
for i, (colname, label, lnst) in enumerate(zip(['medianSpeed', 'maxSpeed'], 
                                         ['median', 'max'],
                                         ['solid','dotted']
                                         )):
    
    meanval = np.nanmean(arr[:,:,i], axis = 0)
    sem = scipy.stats.sem(arr[:,:,i], axis = 0, nan_policy = 'omit')  
    ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., arr[:,:,i].shape[0]-1) 
    ax.fill_between(freqs, 
                    meanval - ci,
                    meanval + ci, 
                    facecolor = clr, 
                    alpha = 0.2)
    
    ax.plot(freqs,
            np.nanmean(arr[:,:,i], axis = 0),
            color = clr,  
            linewidth = 1,
            linestyle = lnst,
            label = label,
            )


    ax.set_xlabel('Stimulation frequency (Hz)')
    ax.set_xlim(5, 50)
    ax.set_ylabel("Speed (cm/s)") #label)
    ax.set_xticks([10,20,30,40,50])
    try:
        ax[i].get_legend().remove()  
    except:
        pass

mod_types = ['linear', 'quadratic']
mod_AIC = pd.read_csv(os.path.join(Config.paths['passiveOpto_output_folder'], f"{yyyymmdd}_mixedEffectsModel_AICRsq_{colname}_v_stimFreq_COMBINED.csv"))
AICs = [mod_AIC['Value'].loc[np.where((mod_AIC['Model']==mod_types[0].capitalize()) & (mod_AIC['Metric']=='AIC'))[0][0]],
        mod_AIC['Value'].loc[np.where((mod_AIC['Model']==mod_types[0].capitalize()) & (mod_AIC['Metric']=='AIC'))[0][0]]]

selected_type = mod_types[np.argmin(AICs)]

mod = pd.read_csv(os.path.join(Config.paths['passiveOpto_output_folder'], f"{yyyymmdd}_mixedEffectsModel_{selected_type}_{colname}_v_stimFreq_COMBINED.csv"))
        
title = 'frequency'
x =0
p_text = title + ' ' + ('*' * (mod['Pr(>|t|)'][x+1] < FigConfig.p_thresholds).sum())
if (mod['Pr(>|t|)'][x+1] < FigConfig.p_thresholds).sum() == 0:
    p_text += "n.s."
ax.text(0.47,
        1, 
        p_text, 
        ha = 'center', 
        color = FigConfig.colour_config['homolateral'][0], 
        transform=ax.transAxes)
            
ax.set_ylim(0,160)
ax.set_yticks(np.linspace(0,160,5,endpoint=True))
# ax[1].set_ylim(60,220)
# ax[1].set_yticks(np.linspace(60,220,5,endpoint=True))
# ax[2].set_ylim(-0.05,1)
#ax[3].set_ylim(0,1)

# plt.tight_layout(w_pad = 2.8, pad = 0, h_pad = 0)

# plt.legend(ncol = 2, loc = 'upper center', bbox_to_anchor=(0.15,1,0.7,0.2))
plt.tight_layout()

lgd = fig.legend([([FigConfig.colour_config['homolateral'][0]],"solid"), 
                ([FigConfig.colour_config['homolateral'][0]],"dotted")], 
                ["median", "max"], 
                handler_map={tuple: AnyObjectHandler()}, 
                loc = 'upper center', bbox_to_anchor=(0.3,0.9,0.7,0.2),
                mode="expand", borderaxespad=0, ncol = 2)

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"MS2_{yyyymmdd}_locomParamsAcrossMice_duration_speed_latency_vs_stimFreq_HEADHEIGHT_meanmaxTOGETHER.svg",
            dpi = 300, 
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight')


 
