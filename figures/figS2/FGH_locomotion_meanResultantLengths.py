import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.stats
import os
import seaborn as sns
from scipy import stats

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

# PER-MOUSE, BUT A RESULT OF THE RANDOM SLOPE MODEL, NOT BETA12

import scipy.stats
from processing import data_loader, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler


appdx = ''

refs = ['lH1', 'lH1', 'lF1']
limbs = ['rH0', 'lF0', 'rF0']
limb_strs = ['hindlimbs', 'homolateral', 'forelimbs']
yyyymmdds = ['2022-08-18', '2023-09-25', None]
cfg_strs = ['passiveOpto', 'mtTreadmill', 'escape']
mice = [Config.passiveOpto_config['mice'], Config.mtTreadmill_config['egr3_ctrl_mice'], None]

# mice_unilateral_inj = Config.injection_config['right_inj_imp'] + Config.injection_config['left_inj_imp'] 
# mouselist = np.intersect1d(Config.passiveOpto_config['mice'], mice_unilateral_inj)
mouseIDlist = []
mrllist = []
setuplist = []
limblist = []
for yyyymmdd, cfg_str, mouselist in zip(yyyymmdds, cfg_strs, mice):
    for ref, limb, limb_str in zip(refs, limbs, limb_strs):
        ### LOAD FULL DATASET TO COMPUTE SPEED PERCENTILES
        if yyyymmdd is not None:
            datafull = data_loader.load_processed_data(dataToLoad = 'strideParams',
                                                       outputDir = Config.paths[f'{cfg_str}_output_folder'],
                                                       yyyymmdd = yyyymmdd,
                                                       limb = ref, 
                                                       appdx = appdx)[0]
            
            mouse_mask = [m in mouselist for m in datafull['mouseID']]
            datafull = datafull[mouse_mask].reset_index(drop=True)
            
        else:
            datafull = pd.read_csv(r"G:\strideParams_escape.csv")
            
        for m in datafull.mouseID.unique():
            datafull_sub = datafull[datafull.mouseID==m].reset_index(drop=True)
        
            phases = datafull_sub[limb]
            
            if yyyymmdd is not None: # not necessary for escape data
                phases_circ = phases*2*np.pi
                phases_circ = np.where(phases_circ<0, phases_circ+(2*np.pi), phases_circ)
            else:
                phases_circ = phases
            
            mrl = utils_math.mean_resultant_length(phases_circ)
            
            mouseIDlist.append(m)
            setuplist.append(cfg_str)
            limblist.append(limb_str)
            mrllist.append(mrl)

resultant_lengths = pd.DataFrame(np.vstack((mouseIDlist, setuplist, limblist, mrllist)).T, 
                                     columns=['mouseID', 'setup', 'limb', 'MRL'])

resultant_lengths['condition'] = resultant_lengths['limb']+ '_' + resultant_lengths['setup']
resultant_lengths['MRL']=resultant_lengths['MRL'].astype(float)


condition_order = ['mtTreadmill', 'passiveOpto', 'escape']
limb_clr_dict = {'hindlimbs': 'homologous', 'homolateral': 'homolateral', 'forelimbs': 'diagonal'}
cond_clr_dict = {'mtTreadmill': 'grey', 'escape': 'black'}

for limb in limb_clr_dict.keys():
    fig, ax = plt.subplots(1,1,figsize = (1.35,1.3))
    scatter_shift=0.15
    boxplot_shift=-0.15
    
    for i, condition in enumerate(condition_order):
        cond = f"{limb}_{condition}"
        
        mrls = resultant_lengths[resultant_lengths.condition==cond]['MRL']
        clr = FigConfig.colour_config[limb_clr_dict[limb]][2]
        if condition in cond_clr_dict.keys():
            clr = cond_clr_dict[condition]
            
        ax.boxplot(mrls, positions = [i+boxplot_shift], 
                    medianprops = dict(color = clr, linewidth = 1, alpha = 0.6),
                    boxprops = dict(color = clr, linewidth = 1, alpha = 0.6), 
                    capprops = dict(color = clr, linewidth = 1, alpha = 0.6),
                    whiskerprops = dict(color = clr, linewidth = 1, alpha = 0.6), 
                    flierprops = dict(mec = clr, linewidth = 1, alpha = 0.6, ms=2))
        ax.scatter(np.repeat(i+scatter_shift, 
                    len(mrls)), 
                    mrls, 
                    color =  clr, alpha = 0.8, s = 5, zorder = 3)
        
        if i>0:
            prev_cond = f"{limb}_{condition_order[i-1]}"
            _, pval = stats.ttest_ind(mrls, 
                                      resultant_lengths[resultant_lengths.condition==prev_cond]['MRL'])
            
            p_text = ('*' * (pval < (np.asarray(FigConfig.p_thresholds))).sum())
            if (pval < (np.asarray(FigConfig.p_thresholds)).sum()) == 0 and not np.isnan(pval):
                p_text = "n.s."
                
            ax.hlines(1, i-1, i-scatter_shift, color='grey')
            ax.text(i-0.5+boxplot_shift, 0.99, p_text, ha='center', color='grey')
            print(f"{cond} x {prev_cond}: {pval}")
            
    ax.set_title(cond.split("_")[0].capitalize())

    # xlabels = [s[0].upper() for s in condition_order]
    xlabels = ['motor', 'passive', 'escape']
    ax.set_xticks(np.arange(len(condition_order)),
                        labels=xlabels,
                        fontsize=5)
    ax.set_ylim(-0.05,1)
    ax.set_yticks([0,0.5,1])
    ax.set_ylabel("Mean resultant length")
    plt.tight_layout()

    filename = f"{limb}_mean_resultant_length.svg"
    fig.savefig(Path(FigConfig.paths['savefig_folder']) / filename, dpi = 300, transparent=True)  



