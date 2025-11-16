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

from processing import injection_side_classifier as isc

# ########## PASSIVE OPTO ########## 
yyyymmdd = '2022-08-18'
appdx = ''
mice_po = np.concatenate((Config.passiveOpto_config['mice'], Config.passiveOpto_config['mice_pilot']))

# ########## rH0 ########## 
features = ['rH0']
limbRef = 'lH1'
df, _, _ = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                           dataToLoad = "strideParams", 
                                           yyyymmdd = yyyymmdd,
                                           appdx = appdx,
                                           limb = limbRef)

df_sub = isc.prepare_data(df, target_col='injection', dataToClassify=features, mice=mice_po)
fprPO_rH, tprPO_rH, _ = isc.classify_injection_side(df_sub, 
                                                    dataToClassify=features, 
                                                    target_col='injection', 
                                                    # crossvalidation=True # pval=0.0000 (takes ages to run)
                                                    )


# ########## rF0 ########## 
features2 = ['rF0']
limbRef2 = 'lF1'
df2, _, _ = data_loader.load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"], 
                                           dataToLoad = "strideParams", 
                                           yyyymmdd = yyyymmdd,
                                           appdx = appdx,
                                           limb = limbRef2)

df_sub2 = isc.prepare_data(df2, target_col='injection', dataToClassify=features2, mice=mice_po)
fprPO_rF, tprPO_rF, _ = isc.classify_injection_side(df_sub2, 
                                                    dataToClassify=features2, 
                                                    target_col='injection', 
                                                    # crossvalidation=True # pval = 0.0000 (takes ages to run)
                                                    )


# ########## y SHUFFLING ON A MERGED DATASET ############
# merge hindlimb and forelimb homologous phase data
df_merged = df.merge(df2, on=['mouseID', 'expDate', 'stimFreq', 'headLVL', 'strideNum'], suffixes=['refHIND', 'refFORE'])
df_merged_sub = df_merged[['mouseID', 'rH0refHIND', 'rF0refFORE']]
df_cleaned = df_merged_sub.rename({"rH0refHIND": "rH0", "rF0refFORE": "rF0"}, axis=1)

features_comb = ['rH0', 'rF0']
df_sub = isc.prepare_data(df_cleaned, target_col='injection', dataToClassify=features_comb, mice=mice_po)

df_shuffle = df_sub.copy()
df_shuffle['injection'] = df_shuffle['injection'].sample(frac=1).reset_index(drop=True)

fprPO_shuffled, tprPO_shuffled, _ = isc.classify_injection_side(df_shuffle, dataToClassify=features_comb, target_col='injection')#, crossvalidation=True)


# ########## PLOTTING ###########
fig, ax = plt.subplots(1,1,figsize=(1.3,1.3))
ax.plot([0,1], [0,1], linestyle='dashed', color='black', lw=0.5)
ax.plot(fprPO_rH, tprPO_rH, color=FigConfig.colour_config['homologous'][3], lw=1.5)
ax.plot(fprPO_rF, tprPO_rF, color=FigConfig.colour_config['diagonal'][2], lw=1.5, ls='dashdot')
ax.plot(fprPO_shuffled, tprPO_shuffled, color=FigConfig.colour_config['greys'][1], lw=1.5, ls='dotted')
ax.set_xlim(0,1)
ax.set_ylim(0,1.1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_xticks([0,0.5,1.0])
ax.set_yticks([0,0.5,1.0])

ax.text(0.05, 1.1, "hind: ***", color=FigConfig.colour_config['homologous'][3], fontsize=5)
ax.text(0.05, 0.95, "fore: ***", color=FigConfig.colour_config['diagonal'][2], fontsize=5)
ax.text(0.65, 1.1, "shuffled", color=FigConfig.colour_config['greys'][1], fontsize=5)
ax.hlines(1.065, 0.06, 0.25, color=FigConfig.colour_config['homologous'][3], lw=1)
ax.hlines(0.915, 0.06, 0.25, color=FigConfig.colour_config['diagonal'][2], lw=1, linestyle='dashdot')
ax.hlines(1.065, 0.66, 1, color=FigConfig.colour_config['greys'][1], lw=1, linestyle='dotted')
ax.set_title("Homologous phase")
plt.tight_layout()

filename = "injectionSideClassifier_homologous_separate.svg"
fig.savefig(Path(FigConfig.paths['savefig_folder']) / filename, dpi = 300, transparent=True)  
    
    
  