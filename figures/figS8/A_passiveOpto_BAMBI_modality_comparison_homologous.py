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
mouselist = ['FAA1034836', 'FAA1034839', 'FAA1034842',
                  'FAA1034868', 'FAA1034869', 'FAA1034942',
                  'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949',
                  'BAA1098955', 'FAA1034469', 'FAA1034471', 'FAA1034570',
               'FAA1034572', 'FAA1034573', 'FAA1034575', 'FAA1034576']

yyyymmdd = '2022-08-18'
yyyymmdd2 = '2022-02-26'
limb = 'rH0'
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
                                            lower_bound=lower_bound1,
                                            flip=True
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
                                            lower_bound=lower_bound2,
                                            flip=True
                                            )

# def flip_right_inj(df, colsToFlip, mouse_col = 'mouse'):
#     # FLIP DATA
#     df_flipped = df.copy()
#     right_inj = np.asarray([m in Config.injection_config["right_inj_imp"] for m in df_flipped["mouse"]])
#     df_flipped.loc[right_inj, colsToFlip] = df_flipped.loc[right_inj, colsToFlip] *-1
#     return df_flipped

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
    # change data scale from (0,2pi] to (-pi,pi]
    cond1[cond1[['x1', 'x2']]>1] = cond1[cond1[['x1', 'x2']]>1]-2
    cond2[cond2[['x1', 'x2']]>1] = cond2[cond2[['x1', 'x2']]>1]-2
    merged[merged[['x1_cond1', 'x2_cond1', 'x1_cond2', 'x2_cond2']]>1] = merged[merged[['x1_cond1', 'x2_cond1', 'x1_cond2', 'x2_cond2']]>1]-2
    
    
    # # flip data to account for injection side variability (already flipped in mp.compute_bimodal_peaks)
    # cond1 = flip_right_inj(cond1, colsToFlip = ['x1', 'x2'])
    # cond2 = flip_right_inj(cond2, colsToFlip = ['x1', 'x2'])
    # merged = flip_right_inj(merged, colsToFlip = ['x1_cond1', 'x2_cond1', 'x1_cond2', 'x2_cond2'])
    
    for index, row in merged.iterrows():
        for comb in combinations:
            if np.isnan(np.asarray((row[comb[0]], row[comb[1]]))).sum() == 0:
                ax[i].plot(
                        [row[comb[0]], row[comb[1]]],
                        [row[comb[0].replace('x','y')], row[comb[1].replace('x','y')]],
                        color = FigConfig.colour_config['greys'][2]
                        )
    
    # plot bimodal or unimodal point for condition 1
    ax[i].scatter(
        np.concatenate((cond1['x1'], cond1['x2'])),
        np.concatenate((cond1['y1'], cond1['y2'])), 
        color = FigConfig.colour_config['homologous'][0],
        s=6
        )
    
    # plot bimodal or unimodal point for condition 2
    ax[i].scatter(
        np.concatenate((cond2['x1'], cond2['x2'])),
        np.concatenate((cond2['y1'], cond2['y2'])), 
        color = FigConfig.colour_config['homologous'][4],
        s=6
        )

    ax[i].set_ylim(0,1.1)
    ax[i].set_yticks([0,0.5,1])
    ax[i].set_xlim(-1.1,1.2)
ax[1].set_xticks([-1,-0.5,0,0.5,1], labels=["-π","-0.5π","0","0.5π","π"])
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
                        color=FigConfig.colour_config['homologous'][0], clip_on=False)
marker2 = patches.Circle((0.56, 0.995), radius=0.01, transform=fig.transFigure, 
                        color=FigConfig.colour_config['homologous'][4], clip_on=False)
fig.patches.append(marker)
fig.patches.append(marker2)

fig.text(0.13, 0.98, "(141,149] deg", fontsize=5, color=FigConfig.colour_config['homologous'][0])
fig.text(0.59, 0.98, "(166,174] deg", fontsize=5, color=FigConfig.colour_config['homologous'][4])

fig.text(0.95, 0.9, "bimodal", fontsize=5, color = 'black', ha='right')
fig.text(0.95, 0.54, "unimodal", fontsize=5, color = 'black', ha='right')

plt.tight_layout()

filename = f"unimodal_vs_bimodal_{limb}_ref{reflimb}_{lower_bound1}_{upper_bound1}_{lower_bound2}_{upper_bound2}.svg"
fig.savefig(Path(FigConfig.paths['savefig_folder']) / filename,
            dpi=300, transparent=True, bbox_inches='tight')