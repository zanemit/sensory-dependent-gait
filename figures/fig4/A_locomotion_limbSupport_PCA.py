import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

fracs = ['frac4','frac3','frac2diag','frac2hmlt','frac2hmlgFORE', 'frac2hmlgHIND','frac1','frac0']
fracs_lbls = ['all 4 limbs', 'any 3 limbs', '2 diagonal', '2 homolateral', 
              '2 forelimbs', '2 hindlimbs', 'any 1 limb', 'none']

## LOADING 
outputDir = Config.paths["passiveOpto_output_folder"]
expDate = '2022-08-18'
predictor = 'snoutBodyAngle'

datafull_path = r"C:\Users\MurrayLab\Documents\PassiveOptoTreadmill\2022-08-18_strideParamsMerged_incline_COMBINEDtrialType_lH1.csv"
datafull = pd.read_csv(datafull_path)
datafull = datafull[datafull['mouseID'].isin(Config.passiveOpto_config["mice"])].copy()

# PCA of LIMB FRACTIONS
from sklearn.decomposition import PCA
pca = PCA()
datafull_fracs = datafull[fracs].copy().dropna()
pca_components = pca.fit_transform(datafull_fracs)
cumsum = np.cumsum(pca.explained_variance_ratio_)
xrange = np.arange(len(fracs))+1
# num_80prc = xrange[cumsum>0.80][0]
# num_98prc = xrange[cumsum>0.98][0]
num_90prc = xrange[cumsum>0.90][0]
pcs_to_plot = num_90prc

colnames = [f"limbSupportPC{i}" for i in range(1,pcs_to_plot+1)]
pca_df = pd.DataFrame(pca_components[:,:pcs_to_plot], index = datafull_fracs.index, columns=colnames)

# # plot cumulative variance explained
# fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5))
# ax.plot(xrange, np.cumsum(pca.explained_variance_ratio_))
# plt.tight_layout()

# # insert into OG dataset
# datafull = datafull.drop(['limbSupportPC1','limbSupportPC2', 'limbSupportPC3', 'limbSupportPC4'],axis=1)
# datafull = datafull.join(pca_df, how='left')
# datafull.to_csv(datafull_path, index=False)

loadings = pca.components_.T

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap",[
                                                FigConfig.colour_config['homolateral'][1],
                                                'white',
                                                FigConfig.colour_config['homologous'][1]
                                                            ])
# cols = [f"PC{i}" for i in range(1,7)]
cols = np.arange(1,pcs_to_plot+1)
loadings_df = pd.DataFrame(loadings[:,:pcs_to_plot], index=fracs, columns=cols)
fig, ax = plt.subplots(1,1,figsize=(2,1.5))
sns.heatmap(loadings_df, cmap=cmap, center=0, ax=ax, 
            vmin=-1, vmax=1, linecolor='grey', linewidths=0.2)
ax.set_yticks(np.arange(len(fracs))+0.5, labels=fracs_lbls)
ax.set_ylim(bottom=ax.get_ylim()[0]+0.05)
ax.set_xlim(right=ax.get_xlim()[1]+0.05)
ax.set_xlabel("PC")

# add explained variance
variances = [f"{var*100:.0f}" for var in pca.explained_variance_ratio_][:pcs_to_plot]
for i, var in enumerate(variances):
    ax.text(i+0.5, ax.get_ylim()[1]-0.2, var, color='#b3b3b3', fontsize=5, ha='center')
ax.text(-3, ax.get_ylim()[1]-0.2, "% var explained", color='#b3b3b3', fontsize=5)

# add a label to the colourbar
cbar = ax.collections[0].colorbar
cbar.set_label("PC loading")

plt.tight_layout()
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], "limbSupport_PCA_heatmap.svg"), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
