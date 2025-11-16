import sys
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import multimodal_processor as mp
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

dir = r"C:\Users\MurrayLab\Documents\BAMBI"

# data from mice with bimodal lF0 distributions ('mu' are modes - peaks)
mouselist = ['FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842',
                  'FAA1034867', 'FAA1034868', 'FAA1034869', 'FAA1034942',
                  'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949',
                  'BAA1098955', 'FAA1034469', 'FAA1034471', 'FAA1034570',
               'FAA1034572', 'FAA1034573', 'FAA1034575', 'FAA1034576']

yyyymmdd = '2022-08-18'
yyyymmdd2 = '2022-02-26'
limb = 'lF0'
reflimb = 'lH1'
pred  ='snoutBodyAngle'
lower_bound1 = 141; upper_bound1 = 149
lower_bound2 = 166; upper_bound2 = 174

unimodal_lowSBA, bimodal_lowSBA = mp.compute_bimodal_peaks(
                                            dir=dir,
                                            yyyymmdd=yyyymmdd,
                                            limb = limb,
                                            reflimb=reflimb,
                                            mouselist=mouselist,
                                            yyyymmdd2=yyyymmdd2,
                                            pred=pred,
                                            upper_bound=upper_bound1,
                                            lower_bound=lower_bound1
                                            )
unimodal_highSBA, bimodal_highSBA = mp.compute_bimodal_peaks(
                                            dir=dir,
                                            yyyymmdd=yyyymmdd,
                                            limb = limb,
                                            reflimb=reflimb,
                                            mouselist=mouselist,
                                            yyyymmdd2=yyyymmdd2,
                                            pred=pred,
                                            upper_bound=upper_bound2,
                                            lower_bound=lower_bound2
                                            )


nrows=2
fig, ax = plt.subplots(nrows, 1, figsize=(1.3,1.25), sharex = True)
for i, mode in enumerate(['bimodal', 'unimodal']):
    cond1, cond2, merged, combinations = mp.process_bimodal_and_unimodal(
                                  df_bimod_cond1 = bimodal_lowSBA, 
                                  df_unimod_cond1 = unimodal_lowSBA, 
                                  df_bimod_cond2 = bimodal_highSBA, 
                                  df_unimod_cond2 = unimodal_highSBA,
                                  mode=mode
                                  )  
        
    
    for index, row in merged.iterrows():
        for comb in combinations:
            if np.isnan(np.asarray((row[comb[0]], row[comb[1]]))).sum() == 0:
                ax[i].plot(
                        [row[comb[0]], row[comb[1]] ],
                        [row[comb[0].replace('x','y')], row[comb[1].replace('x','y')]],
                        color = FigConfig.colour_config['greys'][2]
                        )
    
    # plot bimodal or unimodal point for condition 1
    ax[i].scatter(
        np.concatenate((cond1['x1'], cond1['x2'])),
        np.concatenate((cond1['y1'], cond1['y2'])), 
        color = FigConfig.colour_config['homolateral'][0],
        s=6
        )
    
    # plot bimodal or unimodal point for condition 2
    ax[i].scatter(
        np.concatenate((cond2['x1'], cond2['x2'])),
        np.concatenate((cond2['y1'], cond2['y2'])), 
        color = FigConfig.colour_config['homolateral'][3],
        s=6
        )

    ax[i].set_ylim(0,1.1)
    ax[i].set_yticks([0,0.5,1])
    ax[i].set_xlim(0.2,1.7)
ax[1].set_xticks([0.5,1,1.5], labels=["0.5π", "π", "1.5π"])
fig.text(0.03, 0.6, "Component weight", va='center', ha='center', rotation='vertical')
ax[1].set_xlabel("Peak position (rad)")


import matplotlib.patches as patches
import matplotlib.transforms as transforms
trans = transforms.ScaledTranslation(0.05, -0.1, fig.dpi_scale_trans)
# marker = patches.Circle((0.68, 0.985), radius=0.01, transform=fig.transFigure, 
#                         color=FigConfig.colour_config['homolateral'][3], clip_on=False)
# marker2 = patches.Circle((0.82, 0.985), radius=0.01, transform=fig.transFigure, 
#                         color=FigConfig.colour_config['homolateral'][0], clip_on=False)
# fig.patches.append(marker)
# fig.patches.append(marker2)

# fig.text(0.18, 0.97, "Snout-hump angle:", fontsize=5)
# fig.text(0.70, 0.97, "low", fontsize=5, color=FigConfig.colour_config['homolateral'][3])
# fig.text(0.84, 0.97, "high", fontsize=5, color=FigConfig.colour_config['homolateral'][0])

marker = patches.Circle((0.1, 0.995), radius=0.01, transform=fig.transFigure, 
                        color=FigConfig.colour_config['homolateral'][0], clip_on=False)
marker2 = patches.Circle((0.56, 0.995), radius=0.01, transform=fig.transFigure, 
                        color=FigConfig.colour_config['homolateral'][3], clip_on=False)
fig.patches.append(marker)
fig.patches.append(marker2)

fig.text(0.13, 0.98, "(141,149] deg", fontsize=5, color=FigConfig.colour_config['homolateral'][0])
fig.text(0.59, 0.98, "(166,174] deg", fontsize=5, color=FigConfig.colour_config['homolateral'][3])

fig.text(0.95, 0.9, "bimodal", fontsize=5, color = 'black', ha='right')
fig.text(0.95, 0.53, "unimodal", fontsize=5, color = 'black', ha='right')

plt.tight_layout()

filename = f"unimodal_vs_bimodal_{limb}_ref{reflimb}_{lower_bound1}_{upper_bound1}_{lower_bound2}_{upper_bound2}.svg"
fig.savefig(Path(FigConfig.paths['savefig_folder']) / filename,
            dpi=300, transparent=True, bbox_inches='tight')