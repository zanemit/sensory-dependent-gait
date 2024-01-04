import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch as patch
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

param = 'speed'
appdx = ''
refLimb = 'lH1'

xmin = 10
xmax = 150
interval = 10
xlbl = 'Speed (cm/s)'

x = np.arange(xmin, xmax, interval)

fracs = ['frac4','frac3','frac2diag','frac2hmlt','frac2hmlgFORE', 'frac2hmlgHIND','frac1','frac0']
fracs_lbls = ['4', '3', '2: diagonal', '2: homolateral', '2: homologous fore', '2: homologous hind', '1', '0']

subplot_num = 2
fig, ax = plt.subplots(1,subplot_num, figsize = (2.5,1.5), sharey = True)

for axi, (outputDir, expDate, tlt) in enumerate(zip([Config.paths["passiveOpto_output_folder"],
                                              Config.paths["mtTreadmill_output_folder"]],
                                               ['2022-08-18', '2021-10-23'],
                                               ['Passive treadmill', 'Motorised treadmill'])):
    df, yyyymmdd, limb = data_loader.load_processed_data(outputDir = outputDir,
                                                         dataToLoad = 'supportFractionsMerged', 
                                                         yyyymmdd = expDate,
                                                         appdx = appdx,
                                                         limb = refLimb)
    
    mice = np.unique(df['mouseID'])
    y_arr = np.zeros((mice.shape[0], x.shape[0], len(fracs))) * np.nan
    for im, m in enumerate(mice):
        df_sub = df[df['mouseID'] == m]
        df_sub = df_sub.iloc[np.random.randint(0,df_sub.shape[0], 1000),:]
        for ix, xitem in enumerate(x):
            df_sub2 = df_sub[(df_sub['speed']>=(xitem-(interval/2)))&(df_sub['speed']<(xitem+(interval/2)))]
            y_arr[im, ix, :] = [df_sub2[f].mean() for f in fracs]
    y = np.nanmean(y_arr, axis = 0)
       
    xmean = np.mean(x)
    xstd = np.std(x)
    xtext_pos = [xmean-0.875*xstd, xmean-0.625*xstd, xmean-0.375*xstd, xmean-0.125*xstd,
             xmean+0.125*xstd, xmean+0.375*xstd, xmean+0.625*xstd, xmean+0.875*xstd]
    xint_pos = int(len(x)/2)
    xint_pos = [xint_pos-0.7*xint_pos, xint_pos-0.5*xint_pos, xint_pos+0.3*xint_pos, xint_pos-0.1*xint_pos,
            xint_pos+0.1*xint_pos, xint_pos+0.3*xint_pos, xint_pos+0.5*xint_pos, xint_pos+0.7*xint_pos]
    ytext_pos = []
    xticks = np.linspace(xmin, xmax, 5, endpoint = True)
     
    # Plot
    ax[axi].stackplot(x,
                 y[:,0],
                 y[:,1],
                 y[:,2],
                 y[:,3],
                 y[:,4],
                 y[:,5],
                 y[:,6],
                 y[:,7],
                 colors = FigConfig.colour_config["mains8"])
    
    ax[axi].set_xticks(xticks)
    ax[axi].set_xlabel(xlbl)
    ax[axi].set_title(tlt)

handles = [
    patch(facecolor= clr, label = lbl)
    for lbl, clr in zip(fracs_lbls, FigConfig.colour_config["mains8"])
    ]
lgd = fig.legend(handles = handles, loc = 'upper center', bbox_to_anchor=(0.95,0.38,0.45,0.5),
                mode="expand", borderaxespad=0, ncol = 1)

ax[0].set_ylabel('Fraction of limb support')

fig.tight_layout(pad=0.5)

# fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"MS2_{yyyymmdd}_limbSupports.svg",
#             bbox_extra_artists = (lgd, ), 
#             bbox_inches = 'tight')