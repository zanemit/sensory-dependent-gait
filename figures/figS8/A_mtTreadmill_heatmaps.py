import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os 
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
import seaborn as sns

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

import scipy.stats
from processing import data_loader, utils_math, utils_processing, treadmill_linearGLM
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

from sklearn.decomposition import PCA

appdx=''
outputDir = Config.paths['mtTreadmill_output_folder']
type_rows = {'level': 0, 'slope': 1}
category_cols = {'alt':0, 'asym':1}
fracs = ['frac4','frac3','frac2diag','frac2hmlt','frac2hmlgFORE', 'frac2hmlgHIND','frac1','frac0']
fracs_lbls = ['all 4 limbs', 'any 3 limbs', '2 diagonal', '2 homolateral', 
              '2 forelimbs', '2 hindlimbs', 'any 1 limb', 'none']

fig, ax = plt.subplots(2,2,figsize=(2.5,2.9), sharey=True)

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap",[
                                                FigConfig.colour_config['homolateral'][1],
                                                'white',
                                                FigConfig.colour_config['homologous'][1]
                                                            ])


# subset RH category
for yyyymmdd, trial_type, mice in zip(
        ['2023-09-25', '2022-05-06'], 
        ['level', 'slope'],
        [Config.mtTreadmill_config["egr3_ctrl_mice"], Config.mtTreadmill_config["mice_incline"]]
        ):
    
    datafull_path = os.path.join(outputDir, f"{yyyymmdd}_strideParamsMerged_{appdx}lH1.csv")
    datafull_full = pd.read_csv(datafull_path)
    datafull_full = datafull_full[datafull_full['mouseID'].isin(mice)].copy()

    for category, category_str in zip(
            [['alt'], ['Rlead', 'Llead']],
            ['alternation', 'asymmetry']
            ):
        datafull = datafull_full[(datafull_full['rH0_categorical'].isin(category))].copy()
        print(f"{category_str}: {datafull.shape[0]} strides")
        category = 'asym' if len(category)>1 and 'Llead' in category and 'Rlead' in category else category[0]
        
        pca = PCA()
        datafull_fracs = datafull[fracs].copy().dropna()
        pca_components = pca.fit_transform(datafull_fracs)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        xrange = np.arange(len(fracs))+1
        num_90prc = xrange[cumsum>0.90][0]
        pcs_to_plot = num_90prc
        
        colnames = [f"limbSupportPC{i}" for i in range(1,pcs_to_plot+1)]
        pca_df = pd.DataFrame(pca_components[:,:pcs_to_plot], index = datafull_fracs.index, columns=colnames)
        cols_to_plot = 4 # defined to get equal heatmap cell dims
        
        loadings = pca.components_.T
        cols = np.arange(1,pcs_to_plot+1)
        loadings_df = pd.DataFrame(loadings[:,:pcs_to_plot], index=fracs, columns=cols)
        if loadings_df.shape[1]<cols_to_plot:
            for d in range(loadings_df.shape[1],cols_to_plot):
                loadings_df[d+1] = np.nan
        
        # PLOT
        row = type_rows[trial_type]
        col = category_cols[category]
        divider = make_axes_locatable(ax[row,col])
        cax = divider.append_axes("right", size="8%", pad=0.05)
        show_cbar = col==1
        sns.heatmap(loadings_df, cmap=cmap, center=0, ax=ax[row, col], 
                    vmin=-1, vmax=1, linecolor='grey', linewidths=0.2,
                    cbar_ax=cax if show_cbar else None, cbar=show_cbar)
        
        if not show_cbar:
            cax.set_visible(False)
        ax[row, col].spines['bottom'].set_visible(True)
        ax[row, col].spines['bottom'].set_linewidth(0.2)
        ax[row, col].spines['bottom'].set_color('grey')
        
        ax[row, col].set_yticks(np.arange(len(fracs))+0.5, labels=fracs_lbls)
        ax[row, col].set_ylim(bottom=ax[row, col].get_ylim()[0]+0.05)
        if col==0:
            ax[row, col].set_ylabel(f"{trial_type.capitalize()} trials")
        if row==0:
            ax[row, col].set_title(f"{category_str}")
        if row==1:
            ax[row, col].set_xlim(right=ax[row, col].get_xlim()[1]+0.05)
            ax[row, col].set_xlabel("PC")
        
        # add explained variance
        variances = [f"{var*100:.0f}" for var in pca.explained_variance_ratio_][:pcs_to_plot]
        for i, var in enumerate(variances):
            ax[row, col].text(i+0.5, ax[row, col].get_ylim()[1]-0.2, var, color='#b3b3b3', fontsize=5, ha='center')
        if col==0:
            ax[row, col].text(-4, ax[row, col].get_ylim()[1]-0.2, "% var explained", color='#b3b3b3', fontsize=5)
            if row==0:
                ax[row, col].text(-4, ax[row, col].get_ylim()[1]-0.7, "hindlimb:", color='black', fontsize=6)
        
        cax.set_ylabel("PC loading", rotation=270, labelpad=10, fontsize=6)
        cax.set_yticks([-1.0,-0.5,0,0.5,1.0])
        cax.tick_params(axis='y', length=2, width=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], "limbSupport_PCA_heatmaps_mtTreadmill_supplementary.svg"), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
           
