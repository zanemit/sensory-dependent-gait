import scipy.stats
import numpy as np
import pandas as pd
from pathlib import Path
import itertools

from processing.data_config import Config

def flip_left_inj(df, colsToFlip, mouse_col = 'mouse'):
    # FLIP DATA (all 'mu' cols are initially positive: in range [0,2pi])
    df_flipped = df.copy()
    left_inj = np.asarray([m in Config.injection_config["left_inj_imp"] for m in df_flipped["mouse"]])
    # flip data without changing the scale (stll [0,2pi])
    df_flipped.loc[left_inj, colsToFlip] = -(df_flipped.loc[left_inj, colsToFlip])+(2*np.pi)
    return df_flipped

def compute_bimodal_peaks(dir, yyyymmdd, limb, reflimb, mouselist,pred,
                          yyyymmdd2=None, upper_bound=149, lower_bound=141, flip=False):
    filepath = Path(dir) / f"{yyyymmdd}_BAMBI_{limb}_ref{reflimb}_{pred}_{lower_bound}_{upper_bound}_chains3_iters10000.csv"
    bambi = pd.read_csv(filepath, index_col=0)
    
    if yyyymmdd2 is not None:
        filepath2 = Path(dir) / f"{yyyymmdd2}_BAMBI_{limb}_ref{reflimb}_{pred}_{lower_bound}_{upper_bound}_chains3_iters10000.csv"
        bambi2 = pd.read_csv(filepath2, index_col=0)
        bambi = pd.concat([bambi, bambi2])
    
    bambi = bambi[np.asarray([m in mouselist for m in bambi["mouse"]])].reset_index(drop=True)
    
    if flip:
        bambi = flip_left_inj(bambi, colsToFlip = ['mu1', 'mu2', 'mu3', 'mu4'])
    
    unimodal = bambi[bambi['pmix2'].isna()].reset_index(drop=True)
    multimodal = bambi[~bambi['pmix2'].isna()].reset_index(drop=True)
    bimodal = multimodal[multimodal['pmix3'].isna()].reset_index(drop=True)
    if not np.all(bambi['pmix3'].isna()):
        multimodal = multimodal[~multimodal['pmix3'].isna()].reset_index(drop=True)
        print(f"Initial multimodal component: {multimodal['mouse'].values}")
        
        pmix_multimodal_mask = multimodal[['pmix1','pmix2', 'pmix3']].min(axis=1)>0.2
        true_multimodal = multimodal[pmix_multimodal_mask]
        print(f"Final multimodal component: {true_multimodal['mouse'].values}")
        false_bimodal = multimodal[~pmix_multimodal_mask].reset_index(drop=True)
        
        # find the colname where min value is
        colmin_name = false_bimodal[['pmix1','pmix2', 'pmix3']].idxmin(axis=1)
        colmin_num = [int(s[-1]) for s in colmin_name]
        colnames = ['mu', 'pmix', 'kappa']
        colmin_list = [[f'{name}{val}' for name in colnames] for val in colmin_num]
        
        # replace columns with min pmax with nans
        for i, columns_to_nan in enumerate(colmin_list):
            false_bimodal.loc[i, columns_to_nan] = np.nan  
        bimodal = pd.concat([bimodal, false_bimodal], ignore_index=True)
    
    mu1 = bimodal[['mu1','mu2', 'mu3']].min(axis=1)
    mu2 = bimodal[['mu1','mu2', 'mu3']].max(axis=1)
    
    # remove when one peak is too small or when differences are too small
    pmix_diff_mask = (bimodal[['pmix1','pmix2','pmix3']].min(axis=1)>0.2) & (abs(mu2-mu1)>0.2*np.pi)
    mu1 = mu1[pmix_diff_mask].reset_index(drop=True)
    mu2 = mu2[pmix_diff_mask].reset_index(drop=True)
    
    cluster1_mean = scipy.stats.circmean(mu1, high=2*np.pi, low=0)
    cluster1_sd = scipy.stats.circstd(mu1, high=2*np.pi, low=0)
    cluster2_mean = scipy.stats.circmean(mu2, high=2*np.pi, low=0)
    cluster2_sd = scipy.stats.circstd(mu2, high=2*np.pi, low=0)
    
    print(f"Number of mice with bimodal distributions: {len(mu1)} --- {len(mu1)*100/len(mouselist):.0f}%")
    print(f"These mice are: {bimodal.loc[pmix_diff_mask,'mouse'].values}")
    print(f"Bimodal peak 1: {(cluster1_mean/np.pi):.2f}π, sd {(cluster1_sd/np.pi):.2f}π")
    print(f"Bimodal peak 2: {(cluster2_mean/np.pi):.2f}π, sd {(cluster2_sd/np.pi):.2f}π")
    
    if len(mu1) < bimodal.shape[0]:
        false_unimodal = bimodal[~pmix_diff_mask].reset_index(drop=True)
        colmin_name = false_unimodal[['pmix1','pmix2','pmix3']].idxmin(axis=1)
        colmin_num = [int(s[-1]) for s in colmin_name]
        colnames = ['mu', 'pmix', 'kappa']
        colmin_list = [[f'{name}{val}' for name in colnames] for val in colmin_num]
        for i, columns_to_nan in enumerate(colmin_list):
            false_unimodal.loc[i, columns_to_nan] = np.nan  
        unimodal = pd.concat([unimodal, false_unimodal], ignore_index=True)
    
    mu_uni = unimodal[['mu1','mu2', 'mu3']].min(axis=1)   
    unimodal_mean = scipy.stats.circmean(mu_uni, high=2*np.pi, low=0)
    unimodal_sd = scipy.stats.circstd(mu_uni, high=2*np.pi, low=0)
    print(f"Number of mice with unimodal distributions: {len(mu_uni)} --- {len(mu_uni)*100/len(mouselist):.0f}%")
    print(f"Unimodal peak: {(unimodal_mean/np.pi):.2f}π, sd {(unimodal_sd/np.pi):.2f}π")
    
    bimodal = bimodal[pmix_diff_mask].reset_index(drop=True)
    
    return unimodal, bimodal
        
def get_y_x(df, mode='bimodal'):
    ymax = df[['pmix1','pmix2','pmix3']].max(axis=1)
    ymax_names = df[['pmix1','pmix2','pmix3']].idxmax(axis=1)
    xmax_names = [f"mu{s[-1]}" for s in ymax_names]
    xmax = np.empty(ymax.shape[0])*np.nan
    
    # create 'bimodal' placeholedrs for mode='unimodal'
    xmin_names = np.empty(ymax.shape[0])*np.nan
    xmin = np.empty(ymax.shape[0])*np.nan
    ymin = np.empty(ymax.shape[0])*np.nan
    
    if mode=='bimodal':
        ymin = df[['pmix1','pmix2','pmix3']].min(axis=1)
        ymin_names = df[['pmix1','pmix2','pmix3']].idxmin(axis=1)
        xmin_names = [f"mu{s[-1]}" for s in ymin_names]
            
    for row, (colname_min, colname_max) in enumerate(zip(
                                    xmin_names, xmax_names
                                    )):
        if mode=='bimodal':
            xmin[row] = df.loc[row, colname_min] / np.pi
        xmax[row] = df.loc[row, colname_max] / np.pi
    

    return   pd.DataFrame({
            "mouse": df.mouse,
            "x1": xmin,
            "x2": xmax,
            "y1": ymin,
            "y2": ymax
            })

def process_bimodal_and_unimodal(df_bimod_cond1, df_unimod_cond1, df_bimod_cond2, df_unimod_cond2, mode='bimodal'):
    # get mice that have a bimodal distribution in at least one interval
    mice = np.union1d(df_bimod_cond1.mouse, df_bimod_cond2.mouse)
    if mode=='unimodal':
        unimod_mice = np.union1d(df_unimod_cond1.mouse, df_unimod_cond2.mouse)
        mice = np.setdiff1d(unimod_mice, mice)
    
    bimod_cond1 = get_y_x(df_bimod_cond1, mode='bimodal')
    bimod_cond2 = get_y_x(df_bimod_cond2, mode='bimodal')
    unimod_cond1 = get_y_x(df_unimod_cond1, mode='unimodal')
    unimod_cond2 = get_y_x(df_unimod_cond2, mode='unimodal')
    
    # combine data per condition
    cond1 = pd.concat([bimod_cond1, unimod_cond1], ignore_index=True)
    cond2 = pd.concat([bimod_cond2, unimod_cond2], ignore_index=True)
    
    # filter mice
    cond1 = cond1[[m in mice for m in cond1['mouse']]]
    cond2 = cond2[[m in mice for m in cond2['mouse']]]
    
    # merge dataframes
    merged = pd.merge(cond1, cond2, on='mouse', suffixes = ['_cond1', '_cond2'])
    
    # get combinations to connect with lines
    x_parts = ["x1", "x2"]
    cond_parts = ["cond1", "cond2"]
    combinations = [(f"{x1}_{cond1}", f"{x2}_{cond2}")
                    for x1, x2 in itertools.product(x_parts, repeat=2)
                    for cond1, cond2 in itertools.product(cond_parts, repeat=2)
                    if cond1 != cond2 and cond1 < cond2]
    
    return cond1, cond2, merged, combinations        


          