import sys
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

import scipy.stats
from processing import data_loader, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

yyyymmdd = '2022-08-18'
param = 'snoutBodyAngle'
group_num = 3 #4 #5
plot_num = 2
limbRef = 'lH1'
appdx = ''
clr_str = 'homologous'
lnst = 'solid'

fig, ax = plt.subplots(3, plot_num,  
                       gridspec_kw={'wspace':0.15,'hspace':0,'top':0.95, 'bottom':0.05, 'left':0.05, 'right':0.95}, 
                       figsize = (1*plot_num,1.65*3), subplot_kw = dict(projection = 'polar'))

df, _, _ = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                           dataToLoad = "strideParams", 
                                           yyyymmdd = yyyymmdd,
                                           appdx = appdx,
                                           limb = limbRef)

df = df[np.asarray([m in Config.passiveOpto_config["mice"] for m in df["mouseID"]])]

conditions = np.unique(df[param])

# split data into group_num groups
# data_split = np.asarray([141,147,154,161,167,174])
data_split = np.asarray([141,152,163,174])
# data_split = np.asarray([141,150,158,166,174])

histAcrossMice2 = np.empty((np.unique(df['mouseID']).shape[0], Config.passiveOpto_config["kde_bin_num"]+1, group_num))
histAcrossMice2[:] = np.nan

for row_axid, (dataToPlot, dataToPlot_type, y_maxlim) in enumerate(zip(
                                                ['lF0', 'rF0', 'rH0'],
                                                ['homolateral', 'diagonal', 'homologous'],
                                                [0.70 , 0.5, 0.8]
                                                )):    
    
    means = np.empty((len(np.unique(df['mouseID'])),2)) * np.nan
    for im, m in enumerate(np.unique(df['mouseID'])):
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
            
            # mean
            means[im, i] = utils_math.circular_stats(grouped_dict[gkey], stat = 'mean')
            
            # polar plot
            kde_bins, kde = utils_math.von_mises_kde_exact(grouped_dict[gkey]*2*np.pi, 10, Config.passiveOpto_config["kde_bin_num"])
            kde_bins_points = np.concatenate((kde_bins, [kde_bins[0]]))
            kde_points = np.concatenate((kde, [kde[0]]))
            
            ax[row_axid, i].fill_between(kde_bins_points, np.repeat(0,len(kde_points)), kde_points, facecolor = FigConfig.colour_config[dataToPlot_type][2], alpha = 0.2)
            histAcrossMice2[im,:, i] = kde_points
    
    print(dataToPlot, 
          utils_math.circular_stats(means[:,0], stat = 'mean'), 
          scipy.stats.circstd(means[:,0]*2*np.pi, high = 0.5*2*np.pi, low = -0.5*2*np.pi)/(2*np.pi))   
    print(dataToPlot, 
          utils_math.circular_stats(means[:,1], stat = 'mean'), 
          scipy.stats.circstd(means[:,1]*2*np.pi, high = 0.5*2*np.pi, low = -0.5*2*np.pi)/(2*np.pi))      
    histAcrossMice2_mean = np.nanmean(histAcrossMice2, axis = 0)
    
    for i, gkey in enumerate(keys[[0,group_num-1]]):    
        ax[row_axid, i].spines['polar'].set_visible(False)
        ax[row_axid, i].grid(color = 'lightgrey', linewidth = 0.5)
        titles = [f"{x:.2f}" for x in np.linspace(0,y_maxlim,4, endpoint = True)[1:]]
        ax[row_axid, i].set_rticks([])
        ax[row_axid, i].yaxis.grid(True, color = 'lightgrey', linestyle = 'dashed', linewidth = 0.5)
        ax[row_axid, i].set_thetagrids(angles = (180, 90, 0, 270), labels = (), color = 'black') # gets rid of the diagonal lines
        for a in (180,90,0,270):
            ax[row_axid, i].set_rgrids(np.linspace(0,y_maxlim,4, endpoint = True)[1:], labels = titles, angle = a, fontsize = 6)
        ax[row_axid, i].set_rlim(0,y_maxlim)
        ax[row_axid, i].set_yticks(ax[row_axid, i].get_yticks())
        ax[row_axid, i].set_axisbelow(True)
        ax[row_axid, i].xaxis.set_tick_params(pad=-5)
        if i == 0:
            ax[row_axid, i].set_xticklabels(['π', '0.5π', '', '1.5π'])
            ax[row_axid, i].set_yticklabels(titles, color = 'lightgrey')
        else:
            ax[row_axid, i].set_xticklabels(['','0.5π', '0', '1.5π'])
            ax[row_axid, i].set_yticklabels(["","",""], color = 'lightgrey')

        polar_points = histAcrossMice2_mean[:, i]
        ax[row_axid, i].plot(kde_bins_points, polar_points, color = FigConfig.colour_config[dataToPlot_type][i],linestyle = lnst, linewidth = 1)
    
        ax[0, i].set_title(f'({gkey.left:.0f},{gkey.right:.0f}] deg', size = 6)
    ax[row_axid, 0].text(2.6,y_maxlim + (y_maxlim/0.9),dataToPlot_type,size=6, weight = 'bold')

plt.tight_layout(h_pad = 6)    
plt.text(0.03, 0.1, 'Probability\ndensity', ha = 'center', color = 'lightgrey', size = 6, transform = plt.gcf().transFigure)
plt.text(0.03, 0.92, 'Snout-hump\nangle', ha = 'center', color = 'black', size = 6, transform = plt.gcf().transFigure)

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"MS3_{yyyymmdd}_allPhaseHistogramsAVERAGED_POLAR_{param}{group_num}_ALL_{limbRef}_combinedTrials.svg")
    
         
