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

outputDir = Config.paths['passiveOpto_output_folder']
yyyymmdd = '2022-08-18'

# ------ DETERMINE WHICH PCs ARE RELEVANT ------
# define PC interpretations
pca_interpretations = {
    'head height':{
        'alt':{
            1: '3limb_vs_diagonal',
            2: '4limb_vs_3limb_diagonal',
            3: 'many_vs_few'
            },
        'sync':{
            1: '3limb_vs_hindlimb',
            2: '4limb_vs_3limb_2limb_1limb',
            3: '1limb_vs_3limb_forelimb',
            4: 'diagonal_forelimb_vs_3limb_hindlimb'
            },
        'asym':{
            1: '4limb_vs_3limb',
            2: '3limb_4limb_vs_diagonal',
            3: 'many_vs_few',
            4: '1limb_vs_2limb'
            }
        },
    'surface slope':{
        'alt':{
            1: '3limb_vs_diagonal',
            2: '4limb_vs_3limb_diagonal',
            3: 'many_vs_few'
            },
        'sync':{
            1: '4limb_vs_3limb_2limb_1limb',
            2: '3limb_vs_hindlimb',
            3: '1limb_vs_2limb',
            4: '3limb_hindlimb_vs_diagonal'
            },
        'asym':{
            1: '4limb_vs_3limb',
            2: '3limb_vs_diagonal',
            3: 'many_vs_few',
            4: '1limb_vs_2limb'
            }
        }
    }
pca_interpretation_df = pd.DataFrame([
    (outer, middle, inner, value)
    for outer, middle_dict in pca_interpretations.items()
    for middle, inner_dict in middle_dict.items()
    for inner, value in inner_dict.items()
    ], columns=['trial type', 'rH category', 'pc', 'explanation'])

# load data from models
rows = []
for trial_type in ['head height', 'surface slope']: #<- one extra predictor
    predictor_str = 'speed_snoutBodyAngle' if trial_type=='head height' else 'speed_snoutBodyAngle_incline'
    slopes_str = 'pred2' if trial_type=='head height' else 'pred2pred3'
    interaction = 'TRUE' if trial_type=='head height' else 'TRUEthreeway'
    for rH_category in ['alt', 'asym', 'sync']:
        for pc_num in np.arange(4)+1:
            path = os.path.join(
                outputDir, 
                f"{yyyymmdd}_contCoefficients_mixedEffectsModel_linear_rH{rH_category}_limbSupportPC{pc_num}_vs_{predictor_str}_randSlopes{slopes_str}_interaction{interaction}.csv"
                )
            if os.path.exists(path):
                mod = pd.read_csv(path, index_col=0)
                row = {
                    'trial type': trial_type,
                    'rH category': rH_category,
                    'pc': pc_num,
                    }
                if trial_type=='head height':
                    row['pred2 estimate'] = mod.loc['pred2_centred', 'Estimate']
                    row['pred2 pval'] = mod.loc['pred2_centred', 'Pr(>|t|)']
            
                if trial_type=='surface slope':
                    row['pred3 estimate'] = mod.loc['pred3_centred', 'Estimate']
                    row['pred3 pval'] = mod.loc['pred3_centred', 'Pr(>|t|)']
                    row['pred2:pred3 estimate'] = mod.loc['pred2_centred:pred3_centred', 'Estimate']
                    row['pred2:pred3 pval'] = mod.loc['pred2_centred:pred3_centred', 'Pr(>|t|)']
                rows.append(row)
            else:
                print(f"No such path: {path}\n")
            
df = pd.DataFrame(rows)
df_intermediate = df.merge(pca_interpretation_df, on=['trial type', 'rH category', 'pc'], how='left')
df_intermediate['rH category'] = df_intermediate['rH category'].map({'alt': 'alternation', 'sync': 'synchrony', 'asym': 'asymmetry'})
df_hh = df_intermediate[df_intermediate['pred3 pval'].isna()].copy().dropna(axis=1)
df_slope = df_intermediate[df_intermediate['pred2 pval'].isna()].copy().dropna(axis=1)

df_merged = df_hh.merge(df_slope, on=['rH category', 'explanation'], suffixes=('_hh', '_ss'))

p_threshold = 0.05
mask_hh = df_merged['pred2 pval'] < p_threshold 
mask_slope = (
    (df_merged['pred3 pval'] < p_threshold) |  (df_merged['pred2:pred3 pval'] < p_threshold)
    ) & ((
        np.sign(df_merged['pred2 estimate'])==np.sign(df_merged['pred3 estimate'])
        ) 
            ) 

df_significant = df_merged.loc[mask_hh & mask_slope, :] 

# ------ COMPUTE AND PLOT PCA HEATMAPS ------
fracs = ['frac4','frac3','frac2diag','frac2hmlt','frac2hmlgFORE', 'frac2hmlgHIND','frac1','frac0']
fracs_lbls = ['all 4 limbs', 'any 3 limbs', '2 diagonal', '2 homolateral', 
              '2 forelimbs', '2 hindlimbs', 'any 1 limb', 'none']

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap",[
                                                FigConfig.colour_config['homolateral'][1],
                                                'white',
                                                FigConfig.colour_config['homologous'][1]
                                                            ])

category_cols = {'alt':0, 'sync':1, 'asym':2}
type_rows = {'head height': 0, 'surface slope': 1}
fig, ax = plt.subplots(2,3,figsize=(3.3,2.9), sharey=True)
for appdx, trial_type in zip(['', 'incline_'], ['head height', 'surface slope']):
    datafull_path = os.path.join(outputDir, f"{yyyymmdd}_strideParamsMerged_{appdx}lH1.csv")
    datafull_full = pd.read_csv(datafull_path)
    datafull_full = datafull_full[datafull_full['mouseID'].isin(Config.passiveOpto_config["mice"])].copy()
    
    # subset RH category
    for category, category_str in zip(
            [['alt'], ['sync'], ['Rlead', 'Llead']],
            ['alternation', 'synchrony', 'asymmetry']
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
        show_cbar = col==2
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
        
        # add rectangles if applicable
        sig_PCs = df_significant.loc[df_significant['rH category']==category_str, f"pc_{''.join([s[0] for s in trial_type.split(' ')])}"].values
        if len(sig_PCs)>0:
            for s in sig_PCs:
                rect = Rectangle((s-1, 0), 1, loadings_df.shape[0], fill=False, edgecolor='black', 
                                 lw=1, ls='dashed', clip_on=False)
                ax[row, col].add_patch(rect)
        
        # if category_str=='asymmetry' and trial_type=='head height':
        #     print(loadings_df)
        
plt.tight_layout()
plt.savefig(os.path.join(FigConfig.paths['savefig_folder'], "limbSupport_PCA_heatmaps_supplementary.svg"), 
            dpi = 300, 
            bbox_inches = 'tight',
            transparent = True)
           



