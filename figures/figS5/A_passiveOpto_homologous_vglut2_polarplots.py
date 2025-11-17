import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig


"""
POLARPLOTS: HOMOLOGOUS HINDLIMB
"""
yyyymmdd = '2022-08-18'
param = 'snoutBodyAngle' #'strideLength'
group_num = 3 #4 #5
plot_num = 2
limbRef = 'lH1'
appdx = ''
lnst = 'solid'
dataToPlot = 'rH0'
dataToPlot_type = 'homologous'
# dataToPlot = 'lF0'
# dataToPlot_type = 'homolateral'
y_maxlim = 0.72 

fig, ax = plt.subplots(1, plot_num,  
                       gridspec_kw={'wspace':0.15,'hspace':0,'top':0.95, 'bottom':0.05, 'left':0.05, 'right':0.95}, 
                       figsize = (0.8*plot_num,1.65*0.8), subplot_kw = dict(projection = 'polar'))

df1, _, _ = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                           dataToLoad = "strideParams", 
                                           yyyymmdd = yyyymmdd,
                                           appdx = appdx,
                                           limb = limbRef)

mice_unilateral_inj = Config.injection_config['right_inj_imp'] + Config.injection_config['left_inj_imp'] 
mouselist = np.intersect1d(Config.passiveOpto_config['mice'], mice_unilateral_inj)
df1 = df1[np.asarray([m in mouselist for m in df1["mouseID"]])].reset_index(drop=True)

yyyymmdd2 = '2022-02-26'
df2, _, _ = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                           dataToLoad = "strideParams", 
                                           yyyymmdd = yyyymmdd2,
                                           appdx = appdx,
                                           limb = limbRef)
df2 = df2[np.asarray([m in Config.passiveOpto_config['mice_pilot'] for m in df2["mouseID"]])].reset_index(drop=True)
# mouselist = np.union1d(mouselist, Config.passiveOpto_config['mice_pilot'])

df = df1[['mouseID', 'snoutBodyAngle', dataToPlot]]
df = pd.concat([df, df2[['mouseID', 'snoutBodyAngle', dataToPlot]]], ignore_index=True)

# FLIP DATA
left_inj = np.asarray([m in Config.injection_config["left_inj_imp"] for m in df["mouseID"]])
df.loc[left_inj, dataToPlot] = df.loc[left_inj, dataToPlot] *-1

conditions = np.unique(df[param])

# split data into group_num groups
# data_split = np.asarray([141,147,154,161,167,174])
# data_split = np.asarray([141,152,163,174])
data_split = np.asarray([141,149,166,174])
# data_split = np.asarray([0,2,4,5])

# these mice have unimodal distributions
mouselist = np.asarray([ 'BAA1098955', 'FAA1034469', 'FAA1034471', 'FAA1034570',
       'FAA1034572', 'FAA1034573', 'FAA1034575', 'FAA1034576',
        'FAA1034839',  'FAA1034868', 'FAA1034944', 'FAA1034945',
       'FAA1034947', 'FAA1034949'])

histAcrossMice2 = np.empty((mouselist.shape[0], Config.passiveOpto_config["kde_bin_num"]+1, group_num))
histAcrossMice2[:] = np.nan 
    
means = np.empty((len(mouselist),2)) * np.nan
for im, m in enumerate(mouselist):
    df_sub = df[df['mouseID'] == m]
    df_grouped = df_sub.groupby(pd.cut(df_sub[param], data_split)) 
    group_row_ids = df_grouped.groups
    grouped_dict = {key:df_sub.loc[val,dataToPlot].values for key,val in group_row_ids.items()} 
    keys = np.asarray(list(grouped_dict.keys()))
    for i, gkey in enumerate(keys[[0,group_num-1]]):
        values = grouped_dict[gkey][~np.isnan(grouped_dict[gkey])]
        if values.shape[0] < (Config.passiveOpto_config["stride_num_threshold"]):
            print(f"Mouse {m} data excluded from group {gkey} because there are only {values.shape[0]} valid trials!")
            continue
        
        # polar plot
        kde_bins, kde = utils_math.von_mises_kde_exact(grouped_dict[gkey]*2*np.pi, 10, Config.passiveOpto_config["kde_bin_num"])
        kde_bins_points = np.concatenate((kde_bins, [kde_bins[0]]))
        kde_points = np.concatenate((kde, [kde[0]]))
        
        # KDE peak (mode)
        kde_max = kde_bins[np.argmax(kde)] # to the nearest degree
        means[im, i] = kde_max
        
        ax[i].fill_between(kde_bins_points, np.repeat(0,len(kde_points)), kde_points, facecolor = FigConfig.colour_config[dataToPlot_type][2], alpha = 0.2)
        histAcrossMice2[im,:, i] = kde_points
    
    # from pycircstat.tests import rayleigh
    # values = df_sub[dataToPlot]*2*np.pi
    # values[values<0] = values[values<0]+(2*np.pi)
    # values = values[~np.isnan(values)].to_numpy()
    # p_value, statistic = rayleigh(values)
    # print(f"{m}: {p_value}")

print(dataToPlot, 
      f"{(scipy.stats.circmean(means[:,0], high= 2*np.pi, low=0)/np.pi):.2f}", 
      f"{(scipy.stats.circstd(means[:,0], high = 2*np.pi, low=0)/np.pi):.2f}")    
print(dataToPlot, 
      f"{(scipy.stats.circmean(means[:,1], high= 2*np.pi, low=0)/np.pi):.2f}", 
      f"{(scipy.stats.circstd(means[:,1], high = 2*np.pi, low=0)/np.pi):.2f}")      
histAcrossMice2_mean = np.nanmean(histAcrossMice2, axis = 0)
# histAcrossMice2_mean = scipy.stats.circmean(histAcrossMice2, axis=0, high = 0.5, low=-0.5)
#
for i, gkey in enumerate(keys[[0,group_num-1]]):    
    ax[i].spines['polar'].set_visible(False)
    ax[i].grid(color = 'lightgrey', linewidth = 0.5)
    titles = [f"{x:.2f}" for x in np.linspace(0,y_maxlim,4, endpoint = True)[1:]]
    ax[i].set_rticks([])
    ax[i].yaxis.grid(True, color = 'lightgrey', linestyle = 'dashed', linewidth = 0.5)
    ax[i].set_thetagrids(angles = (180, 90, 0, 270), labels = (), color = 'black') # gets rid of the diagonal lines
    for a in (180,90,0,270):
        ax[i].set_rgrids(np.linspace(0,y_maxlim,4, endpoint = True)[1:], labels = titles, angle = a, fontsize = 6)
    ax[i].set_rlim(0,y_maxlim)
    ax[i].set_yticks(ax[i].get_yticks())
    ax[i].set_axisbelow(True)
    ax[i].xaxis.set_tick_params(pad=-5)
    if i == 0:
        ax[i].set_xticklabels(['π', '0.5π', '', '1.5π'])
        # ax[i].set_yticklabels(titles, color = 'lightgrey')
    else:
        ax[i].set_xticklabels(['','0.5π', '0', '1.5π'])
        # ax[i].set_yticklabels(["","",""], color = 'lightgrey')
    ax[i].set_yticklabels(titles, color = 'lightgrey')

    polar_points = histAcrossMice2_mean[:, i]
    ax[i].plot(kde_bins_points, polar_points, color = FigConfig.colour_config[dataToPlot_type][0],linestyle = lnst, linewidth = 1)

    ax[i].set_title(f'({gkey.left:.0f},{gkey.right:.0f}] deg', size = 6)
# ax[0].text(2.6,y_maxlim + (y_maxlim/0.9),dataToPlot_type,size=6, weight = 'bold')

plt.tight_layout(h_pad = 6)    
plt.text(0.05, 0.15, 'Probability\ndensity', ha = 'center', color = 'lightgrey', size = 6, transform = plt.gcf().transFigure)
plt.text(0.02, 0.815, 'Snout-\nhump\nangle', ha = 'center', color = 'black', size = 6, transform = plt.gcf().transFigure)

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"thesis_{yyyymmdd}_allPhaseHistogramsAVERAGED_POLAR_{param}{group_num}_ALL_{limbRef}_combinedTrials.svg",
            dpi=300,
            transparent=True)
    