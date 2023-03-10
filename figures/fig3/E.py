import sys
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader, utils_math
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

limb = 'lF0'
limbRef = 'lH1'
param = 'snoutBodyAngle'
data_split = [141,147,154,161,167,174]
kde_bin_num = 360
colnum = 2

fig, ax = plt.subplots(1, colnum, figsize=(1.65*colnum, 1.4), subplot_kw = dict(projection = 'polar'))

for k, tlt in enumerate(['level trials', 'incline trials']):
    ax[k].spines['polar'].set_visible(False)
    ax[k].grid(color = 'lightgrey', linewidth = 0.5)
    titles = [f"({data_split[x]},{data_split[x+1]}]" for x in range(len(data_split)-1)]
    ax[k].set_rticks([])
    ax[k].yaxis.grid(True, color = 'lightgrey', linestyle = 'dashed', linewidth = 0.5)
    ax[k].set_thetagrids(angles = (180, 90, 0, 270), labels = (), color = 'black') # gets rid of the diagonal lines
    for a in (180,90,0,270):
        ax[k].set_rgrids(np.linspace(0,len(data_split)-1,len(data_split), endpoint = True)[1:], labels = "", angle = a, fontsize = 6)
    ax[k].set_rlim(0,len(data_split)-1)
    ax[k].set_yticks(ax[k].get_yticks())
    ax[k].set_axisbelow(True)
    ax[k].xaxis.set_tick_params(pad=-5)
    # ax[k].set_yticklabels(titles, color = 'lightgrey')
    ax[k].set_xticklabels(['π','0.5π', '0', '1.5π'])
    ax[k].set_title(tlt)

distributions = np.zeros((4, len(data_split)-1, 20)) * np.nan
for i_data, (k, dir_str, yyyymmdd, appdx, clr) in enumerate(zip([0,1,0,1],
                                   [Config.paths["passiveOpto_output_folder"], Config.paths["passiveOpto_output_folder"], Config.paths["mtTreadmill_output_folder"], Config.paths["mtTreadmill_output_folder"]],
                                   ['2022-08-18', '2022-08-18', '2021-10-23', '2022-05-06'],
                                   ['', '_incline', '', ''],
                                   ['homolateral', 'homolateral', 'greys', 'greys'],
                                   )):
    df, _, _ = data_loader.load_processed_data(outputDir = dir_str, 
                                                dataToLoad = "strideParams", 
                                                yyyymmdd = yyyymmdd,
                                                appdx = appdx,
                                                limb = limbRef)
              
    mice = np.unique(df['mouseID'])
    group_num = len(data_split)-1
    
    x = np.arange(group_num)+1  # radii  
    arr = np.zeros((group_num, mice.shape[0]))*np.nan
        
    for im, m in enumerate(mice):
        df_sub = df[df['mouseID'] == m]
        df_grouped = df_sub.groupby(pd.cut(df_sub[param], data_split)) 
        group_row_ids = df_grouped.groups
        grouped_dict = {key:df_sub.loc[val,limb].values for key,val in group_row_ids.items()} 
        keys = list(grouped_dict.keys())
        for i, gkey in enumerate(keys):
            values = grouped_dict[gkey][~np.isnan(grouped_dict[gkey])]
            if values.shape[0] < (Config.passiveOpto_config["stride_num_threshold"]):
                print(f"Mouse {m} data excluded from group {gkey} because there are only {values.shape[0]} valid trials!")
                arr[i, im] = np.nan
                continue
            
            # polar data
            kde_bins, kde = utils_math.von_mises_kde_exact(grouped_dict[gkey]*2*np.pi, 10, kde_bin_num)
            kde_max = kde_bins[np.argmax(kde)] # to the nearest degree
            arr[i, im] = kde_max
        # ax.plot(arr[:, im],
        #         x, 
        #         color = FigConfig.colour_config[clr][clr_id],
        #         linewidth = 1,
        #         alpha = 0.2) 
        
    distributions[i_data, :, :arr.shape[1]] = arr
    
    y = np.zeros(group_num) * np.nan
    y_std = np.zeros(group_num) * np.nan

    for row in range(arr.shape[0]):
        y[row] = scipy.stats.circmean(arr[row,:], nan_policy = 'omit')
        y_std[row] = scipy.stats.circstd(arr[row,:], nan_policy = 'omit')
    a_plus = y+y_std
    a_minus = (y-y_std)[::-1]
    angles = np.concatenate((a_plus, np.linspace(a_plus[-1], a_minus[0], 1000), a_minus, np.linspace(a_minus[-1], a_plus[0], 1000)))
    radii = np.concatenate((x, np.repeat(5,1000), x[::-1], np.repeat(1,1000)))
        
    ax[k].fill_between(angles, 
                    radii,
                    facecolor = FigConfig.colour_config[clr][2],
                    linewidth = 1,
                    alpha = 0.2)    
        
    ax[k].plot(y,
            x, 
            color = FigConfig.colour_config[clr][2],
            linewidth = 2,
            zorder = 3)  

from matplotlib.lines import Line2D
for i in range(len(titles)):
    plt.text(0.51, 0.375-(i*0.07), titles[i], ha = 'center',  color = 'lightgrey', size = 6, transform = plt.gcf().transFigure)
    fig.add_artist(Line2D([0.31+(i*0.01),0.44], [0.48-(i*0.08),0.39-(i*0.07)], lw=0.5, color='lightgrey', alpha=0.6, transform = plt.gcf().transFigure))
    fig.add_artist(Line2D([0.588, 0.71-(i*0.009)], [0.39-(i*0.07),0.48-(i*0.08)], lw=0.5, color='lightgrey', alpha=0.6, transform = plt.gcf().transFigure))
plt.text(0.51, 0.375-(5*0.07), "Snout-hump angle (deg)", ha = 'center',  color = 'lightgrey', size = 6, transform = plt.gcf().transFigure)

datasets = ['head-fixed-level', 'head-fixed-incline', 'head-free-level', 'head-free-incline']
p_thr = np.asarray(FigConfig.p_thresholds)/5
d_id = [(0,2), (1,3)]
for inum, (i, angles, add_num) in enumerate(zip(d_id,
                                       [[4, 3.5, 3.2, 2.9, 2.5], [3.3, 3.5, 3.1, 2.6, 2.4]],
                                       [[0.5, 1, 1.9, 0.2, 0], [0.5, 0.8, 1.1, 0.2, 0.8]])):
    print(f"Compare {datasets[i[0]]} and {datasets[i[1]]}...")
    for g in range(len(data_split)-1):
        p = float(utils_math.WatsonU2_TwoTest(distributions[i[0],g,:], distributions[i[1],g,:])['p'][1:])
        p_text =  '*' * (p <= p_thr).sum()
        if (p <= p_thr).sum() == 0:
            p_text += "n.s." 
        
        if not (inum == 1 and g == 0):
            ax[inum].text(angles[g], 
                          g+1+add_num[g], 
                          p_text, 
                          color = 'grey')
            

lgd = fig.legend([([FigConfig.colour_config['homolateral'][2]],"solid"), 
                ([FigConfig.colour_config['greys'][2]],"dashed")], 
                ["head-fixed", "head-free"], 
                handler_map={tuple: AnyObjectHandler()}, 
                loc = 'upper center', bbox_to_anchor=(0.15,-0.25,0.7,0.2),
                mode="expand", borderaxespad=0, ncol = 2)
        # print(f"{g}: {p}, {p_text}")
        
fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_phaseAcrossDeg_{limb}_reflimb_{limbRef}.svg",
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight')


