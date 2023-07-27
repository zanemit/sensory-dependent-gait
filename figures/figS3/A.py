import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

nrow = 2; ncol = 2
fig, ax = plt.subplots(1, 3,  
                       gridspec_kw={'wspace':0.15,'hspace':0,'top':0.95, 'bottom':0.05, 'left':0.05, 'right':0.95}, 
                       figsize = (2.2,1.6), subplot_kw = dict(projection = 'polar'))

idealGaitDict = pickle.load(open(Path(Config.paths["passiveOpto_output_folder"]) / "idealisedGaitDict.pkl", "rb" ))

legend_colours = np.empty((3, 0)).tolist()
legend_linestyles = np.empty((3, 0)).tolist()
i = 0; j = 0
for ilimb, limb in enumerate(['homologous', 'homolateral', 'diagonal']):
    for ig, (gait, lnst) in enumerate(zip(['trot', 'bound', 'transverse_gallop_R'],
                          ['solid', 'dotted', 'dashed'])):
        phases = idealGaitDict[gait][limb]
        
        kde_bins, kde = utils_math.von_mises_kde_exact(phases*2*np.pi, 10, Config.passiveOpto_config["kde_bin_num"])
        kde_bins_points = np.concatenate((kde_bins, [kde_bins[0]]))
        kde_points = np.concatenate((kde, [kde[0]]))
        
        ax[ilimb].fill_between(kde_bins_points, 
                             np.repeat(0,len(kde_points)), 
                             kde_points, 
                             facecolor = FigConfig.colour_config[limb][2], 
                             alpha = 0.2,
                             linestyle = lnst,
                             linewidth = 1)
        
        ax[ilimb].plot(kde_bins_points, 
                     kde_points, 
                     color = FigConfig.colour_config[limb][2],
                     linestyle = lnst, 
                     linewidth = 1)
    
        legend_colours[ig].append(FigConfig.colour_config[limb][2])
        if ilimb == 0:
            legend_linestyles[ig] = lnst
        
        ax[ilimb].spines['polar'].set_visible(False)
        ax[ilimb].grid(color = 'lightgrey', linewidth = 0.5)
        titles = ["", "", ""]
        ax[ilimb].set_rticks([])
        ax[ilimb].yaxis.grid(True, color = 'lightgrey', linestyle = 'dashed', linewidth = 0.5)
        ax[ilimb].set_thetagrids(angles = (180, 90, 0, 270), labels = (), color = 'black') # gets rid of the diagonal lines
        for a in (180,90,0,270):
            ax[ilimb].set_rgrids(np.linspace(0,0.8,4, endpoint = True)[1:], labels = titles, angle = a, fontsize = 6)
        ax[ilimb].set_rlim(0,1)
        ax[ilimb].set_yticks(ax[ilimb].get_yticks())
        ax[ilimb].set_axisbelow(True)
        ax[ilimb].xaxis.set_tick_params(pad=-5)
        ax[ilimb].set_title(limb, color = FigConfig.colour_config[limb][2])
    
        if ilimb == 0:
            ax[ilimb].set_xticklabels(['π', '0.5π', '', '1.5π'])
        elif ilimb == 1:
            ax[ilimb].set_xticklabels(['', '0.5π', '', '1.5π'])
        elif ilimb == 2:
            ax[ilimb].set_xticklabels(['', '0.5π', '0', '1.5π'])
    
    i = (ilimb+1) // ncol
    j = (ilimb+1) % ncol

legend_actors = [(legend_colours[i], legend_linestyles[i]) for i in range(len(legend_linestyles))]

# ax[1,1].remove()

lgd = fig.legend(legend_actors, 
                ['trot', 'bound','transverse\ngallop'],
                handler_map={tuple: AnyObjectHandler()}, loc = 'upper left',
                bbox_to_anchor=(0.2,-0.1,0.6,0.3), mode="expand", 
                borderaxespad=0.1, ncol = 2, frameon = True,
                handlelength = 1.5, title = 'Idealised gait')    

fig.savefig(Path(FigConfig.paths['savefig_folder']) / "MS_idealised_gaits.svg",
            dpi = 300, 
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight')

     
