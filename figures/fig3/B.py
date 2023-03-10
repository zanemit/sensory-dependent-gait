import sys
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader, utils_math
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig

param = 'lF0'
group_num = 5
bin_num = 20
dataToPlot = 'snoutBodyAngle'
dataToPlot_type = 'homolateral'
limbRef = 'lH1'
appdx = ''

fig, ax = plt.subplots(1, group_num-1, sharex = True, sharey = True, figsize = (0.7*group_num,1.5)) #(1*gn, 1.5)

for yyyymmdd, mouse_str, clr_str in zip(['2021-10-23', '2022-05-06'],
                               ['mice_level', 'mice_incline'],
                               ['greys', 'homolateral']):
    df, _, _ = data_loader.load_processed_data(outputDir = Config.paths["mtTreadmill_output_folder"], 
                                               dataToLoad = "strideParams", 
                                               yyyymmdd = yyyymmdd,
                                               appdx = appdx,
                                               limb = limbRef)
    
    mice =    Config.mtTreadmill_config[mouse_str] 
    df = df[np.asarray([m in mice for m in df["mouseID"]])]
    data_split = np.linspace(-0.0000001, 1, group_num+1)
    
    axj = 0
    histAcrossMice = np.empty((len(mice), bin_num, group_num))
    histAcrossMice[:] = np.nan
        
    for im, m in enumerate(mice):
        df_sub = df[df['mouseID'] == m]
        df_sub[param] = [x+1 if x<0 else x for x in df_sub[param]]
        df_grouped = df_sub.groupby(pd.cut(df_sub[param], data_split)) 
        group_row_ids = df_grouped.groups
        grouped_dict = {key:df_sub.loc[val, dataToPlot].values for key,val in group_row_ids.items()} 
        keys = np.delete(list(grouped_dict.keys()),4)
        # keys = list(grouped_dict.keys())
        axk = 0
        for i, gkey in enumerate(keys):
            # values = np.nan_to_num(grouped_dict[gkey])
            values = grouped_dict[gkey][~np.isnan(grouped_dict[gkey])]
            if values.shape[0] < Config.mtTreadmill_config['stride_num_threshold']:
                print(f"Mouse {m} excluded from group {gkey}, limb {param} because there are only {values.shape[0]} valid trials!")
                axk = (axk + 1) % group_num
                continue
    
            n, bins, patches = ax[axk].hist(grouped_dict[gkey], 
                                            bins = bin_num, 
                                            color = FigConfig.colour_config[clr_str][i], 
                                            range = [140,180],
                                            density = True, 
                                            alpha = 0.1)
            
            bins_mean = bins[:-1] + np.diff(bins)
            histAcrossMice[im,:, i] = n
            axk = (axk + 1) % group_num
         
    histAcrossMice_mean = np.nanmean(histAcrossMice, axis = 0)
    
    for i, gkey in enumerate(keys):
        total = histAcrossMice_mean[:, axj].sum()
        bins_points = np.concatenate(([bins[0]], np.repeat(bins[1:-1],2), [bins[-1]]))
        histo_points = np.repeat(histAcrossMice_mean[:, axj],2)
    
        ax[axj].plot(bins_points, histo_points, color = FigConfig.colour_config[clr_str][i], linewidth = 2)
        
        if i == 0:
            ax[axj].set_title(f'[0,{2*keys[i].right:.1f}π]')
        # elif i == len(keys)-1:
        #     ax[axj].set_title(f'({2*keys[i].left:.1f}π,2π]')
        else:
            ax[axj].set_title(f'({2*keys[i].left:.1f}π,{2*keys[i].right:.1f}π]')
        
        ax[axj].set_xlim(140,180)
        ax[axj].set_xticks([140,150,160,170,180])
        ax[axj].set_xticklabels(['140','','160','','180'])
    
        axj = (axj + 1) % group_num

ax[0].set_ylabel("Probability density")
ax[0].set_ylim(0,0.18)
ax[0].set_yticks([0,0.06,0.12,0.18])
fig.text(0.5, 0, "Snout-hump angle (deg)", ha = 'center')
plt.tight_layout(w_pad = 1.5)

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_allPhaseHistogramsAVERAGED_{param}{group_num}_allMICE_ref{limbRef}.svg", dpi = 300)
