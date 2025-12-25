from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

from processing import injection_side_classifier as isc

def plot_figS2M():
    # ########## PASSIVE OPTO ########## 
    yyyymmdd = '2023-09-25'
    appdx = ''
    mice_mt = Config.mtTreadmill_config['egr3_ctrl_mice']
    
    # ########## rH0 ########## 
    features = ['rH0']
    limbRef = 'lH1'
    dfMT, _, _ = data_loader.load_processed_data(outputDir = Config.paths["mtTreadmill_output_folder"], 
                                               dataToLoad = "strideParams", 
                                               yyyymmdd = yyyymmdd,
                                               appdx = appdx,
                                               limb = limbRef)
    
    df_sub = isc.prepare_data(dfMT, 
                              target_col='injection', 
                              dataToClassify=features, 
                              mice=mice_mt)
    fprMT_rH, tprMT_rH, accuracyHIND, _ = isc.classify_injection_side(df_sub, 
                                                        dataToClassify=features, 
                                                        target_col='injection', 
                                                        # crossvalidation=True # Permutation p-value: 0.0200 (takes ages to run)
                                                        )
    
    
    # ########## rF0 ########## 
    features2 = ['rF0']
    limbRef2 = 'lF1'
    df2MT, _, _ = data_loader.load_processed_data(outputDir = Config.paths["mtTreadmill_output_folder"], 
                                               dataToLoad = "strideParams", 
                                               yyyymmdd = yyyymmdd,
                                               appdx = appdx,
                                               limb = limbRef2)
    
    df_sub2 = isc.prepare_data(df2MT, 
                               target_col='injection', 
                               dataToClassify=features2, 
                               mice=mice_mt)
    fprMT_rF, tprMT_rF, accuracyFORE, _ = isc.classify_injection_side(df_sub2, 
                                                        dataToClassify=features2, 
                                                        target_col='injection', 
                                                        # crossvalidation=True # Permutation p-value: 0.0200 (takes ages to run)
                                                        )
    
    
    # ########## y SHUFFLING ON A MERGED DATASET ############
    # merge hindlimb and forelimb homologous phase data
    df_merged = dfMT.merge(df2MT, 
                         on=['mouseID', 'expDate', 'trialNum', 'trialType', 'stimType', 'strideNum'], 
                         suffixes=['refHIND', 'refFORE'])
    df_merged_sub = df_merged[['mouseID', 'rH0refHIND', 'rF0refFORE']]
    df_cleaned = df_merged_sub.rename({"rH0refHIND": "rH0", "rF0refFORE": "rF0"}, axis=1)
    
    features_comb = ['rH0', 'rF0']
    df_sub = isc.prepare_data(df_cleaned, target_col='injection', dataToClassify=features_comb, mice=mice_mt)
    
    df_shuffle = df_sub.copy()
    shuffled_accuracies = []
    num_permutations=1000
    for i in range(num_permutations):
        print(f"PERMUTATION: {i+1}")
        df_shuffle['injection'] = df_shuffle['injection'].sample(frac=1, random_state=i).reset_index(drop=True)
    
        fprPO_shuffled, tprPO_shuffled, accuracy_shuffled, _ = isc.classify_injection_side(df_shuffle, dataToClassify=features_comb, target_col='injection')#, crossvalidation=True)
        shuffled_accuracies.append(accuracy_shuffled)
    
    # statistical significance
    hind_p =  ((np.asarray(shuffled_accuracies)>accuracyHIND).sum()+1)/(num_permutations+1)
    print(f"HIND p: {hind_p}")
    hind_p = (hind_p<np.asarray(FigConfig.p_thresholds)).sum()
    hind_p_str = 'n.s.' if hind_p==0 else '*'*hind_p
    
    fore_p =  ((np.asarray(shuffled_accuracies)>accuracyFORE).sum()+1)/(num_permutations+1)
    print(f"FORE p: {fore_p}")
    fore_p = (fore_p<np.asarray(FigConfig.p_thresholds)).sum()
    fore_p_str = 'n.s.' if fore_p==0 else '*'*fore_p
    
    # ########## PLOTTING ###########
    fig, ax = plt.subplots(1,1,figsize=(1.3,1.3))
    ax.plot([0,1], [0,1], linestyle='dashed', color='black', lw=0.5)
    ax.plot(fprMT_rH, tprMT_rH, color=FigConfig.colour_config['homologous'][3], lw=1.5)
    ax.plot(fprMT_rF, tprMT_rF, color=FigConfig.colour_config['diagonal'][2], lw=1.5, ls='dashdot')
    ax.plot(fprPO_shuffled, tprPO_shuffled, color=FigConfig.colour_config['greys'][1], lw=1.5, ls='dotted')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xticks([0,0.5,1.0])
    ax.set_yticks([0,0.5,1.0])
    
    ax.text(0.05, 1.1, f"hind: {hind_p_str}", color=FigConfig.colour_config['homologous'][3], fontsize=5)
    ax.text(0.05, 0.95, f"fore: {fore_p_str}", color=FigConfig.colour_config['diagonal'][2], fontsize=5)
    ax.text(0.65, 1.1, "shuffled", color=FigConfig.colour_config['greys'][1], fontsize=5)
    ax.hlines(1.065, 0.06, 0.25, color=FigConfig.colour_config['homologous'][3], lw=1)
    ax.hlines(0.915, 0.06, 0.25, color=FigConfig.colour_config['diagonal'][2], lw=1, linestyle='dashdot')
    ax.hlines(1.065, 0.66, 1, color=FigConfig.colour_config['greys'][1], lw=1, linestyle='dotted')
    ax.set_title("Homologous phase")
    plt.tight_layout()
    
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    filename = "injectionSideClassifier_homologous_separate_MT.svg"
    savepath = Path(FigConfig.paths['savefig_folder']) / filename
    fig.savefig(savepath, dpi = 300, transparent=True)  
    print(f"FIGURE SAVED AT {savepath}")
      
if __name__=="__main__":  
    plot_figS2M()
        
      