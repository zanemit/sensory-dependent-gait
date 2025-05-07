import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, forceplate_data_manager
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

appdx = 'egr3'
stats_days = [30,70,110]

def get_stats(first_sample, daylist, second_sample=None, window=2):
    pvals = []; ptexts = []
    for d in daylist:
        selected = first_sample.loc[:, (d-window):(d+window)]
        selected = np.asarray(selected).flatten()
        selected = selected[~np.isnan(selected)]
        print(f"first sample mean: {np.mean(selected)}")
        
        if np.all(selected==0):
            pval=np.inf
        elif second_sample is None:
            _, pval = scipy.stats.ttest_1samp(selected, popmean=0)
        else:
            selected_second = second_sample.loc[:, (d-window):(d+window)]
            selected_second = np.asarray(selected_second).flatten()
            selected_second = selected_second[~np.isnan(selected_second)]
            print(f"second sample mean: {np.mean(selected_second)}")
            _, pval = scipy.stats.ttest_ind(selected, selected_second)
        ptxt = ('*' * (pval < np.asarray(FigConfig.p_thresholds)).sum())
        if (pval < np.asarray(FigConfig.p_thresholds)).sum() == 0:
            ptxt += "n.s."
        pvals.append(pval)
        ptexts.append(ptxt)
    return pvals, ptexts
        

"""
PLOTTING ATAXIC PROPERTIES
"""
path = r"C:\Users\MurrayLab\Documents\PassiveOptoTreadmill\Egr3_scoresheet.csv"
props = list(FigConfig.ataxia_score_max.keys())

scores = pd.read_csv(path, index_col = [0,1], header = [0,1])

exp_mice = Config.openField_config["mice"][appdx]
mice = list(Config.openField_config["mouse_procedures"][appdx].keys())

days = sorted(np.unique(scores.columns.get_level_values(0)).astype(int))
# mice = [("").join(x.split(" ")[0].split("-")[:2]) for x in np.unique(scores.index.get_level_values(0))]

# PLOT EXP
col_exp = '#4CA7A9'
rows = 3
cols = 3
fig, ax = plt.subplots(rows, cols,figsize = (1.8*cols,1.3*rows))

i = 0; j = 0
for ip, prop in enumerate(props):
    df = pd.DataFrame(np.empty((len(exp_mice), len(days)))*np.nan, index = exp_mice, columns = days)
    for d in days:
        df.loc[:,d] = np.asarray(scores.loc[:,(str(d), prop)] )
    for m in exp_mice:
        df_sub = df.loc[m, :]
        ax[i,j].plot(df.columns, 
                 df_sub,
                 color = col_exp,
                 alpha =0.3
                 )
    
    df_sub = df.loc[exp_mice, :]    
    ax[i,j].plot(df.columns,
             np.nanmean(df_sub, axis = 0),
             color = col_exp,
             lw = 1.5)
    
    ymax = FigConfig.ataxia_score_max[prop] -1 # minus one because 0 included  
    ax[i,j].set_xlim(20,120)
    ax[i,j].set_xticks([20,70,120])
    ax[i,j].set_ylim(0,ymax)
    ax[i,j].set_yticks(np.linspace(0,ymax, ymax+1, endpoint = True), 
                  labels = FigConfig.ataxia_explanations[prop])

    ax[i,j].set_xlabel("Age (days)")
    prop = "Head posture" if prop=="Nose down" else prop
    prop = "Trunk posture" if prop=="Belly drag" else prop
    ax[i,j].set_title(prop)#, bbox = dict(pad = 0.1, fc = '#ffcc00', ec = 'none'))
    
    # STATS
    ysubtract = 0.5 if ymax==3 else 0.25
    pvals, ptexts = get_stats(df_sub, stats_days, second_sample=None) 
    for d, p in zip(stats_days, ptexts):
        ax[i,j].text(d, ymax-ysubtract, p, fontsize=5, ha='center')
    
    print(f"{prop}:\nDay {stats_days[0]}: {ptexts[0]}\nDay {stats_days[1]}: {ptexts[1]}\nDay {stats_days[2]}: {ptexts[2]}")
    
    i = (ip+1)//cols
    j = (ip+1)%cols


"""
PLOTTING WEIGHT
"""
path = r"C:\Users\MurrayLab\Documents\PassiveOptoTreadmill\metadataPyRAT_processed.csv"

mouseweights = pd.read_csv(path, index_col = [0,1,2], header = [0,1]).droplevel(level = 0)

ctrl_mice = np.setdiff1d(mice,exp_mice)

mice_f = mouseweights.index.get_level_values(0)[np.where(mouseweights.index.get_level_values(1) == 'f')[0]]
mice_m = mouseweights.index.get_level_values(0)[np.where(mouseweights.index.get_level_values(1) == 'm')[0]]

exp_mice_f = np.intersect1d(mice_f, exp_mice)
exp_mice_m = np.intersect1d(mice_m, exp_mice)
ctrl_mice_f = np.intersect1d(mice_f, ctrl_mice)
ctrl_mice_m = np.intersect1d(mice_m, ctrl_mice)

min_days = np.nanmin(mouseweights.loc[mice].iloc[:, 1::2])
max_days = np.nanmax(mouseweights.loc[mice].iloc[:, 1::2])

dayrange = np.arange(min_days, max_days+1).astype(int)

weight_df = pd.DataFrame(np.empty((len(mice), dayrange.shape[0]))*np.nan, columns = dayrange, index = mice)

for m in mice:
    days = np.asarray(mouseweights.loc[m].iloc[:,1::2])
    weights = np.asarray(mouseweights.loc[m].iloc[:,::2])
    days = days[~np.isnan(days)].astype(int)
    weights = weights[~np.isnan(weights)]
    weight_df.loc[m, days] = weights

weight_df.interpolate(axis = 1, inplace=True)   

clr_dict = {'exp': '#4CA7A9', 'ctrl': 'grey'} #'#f1c232'<- yellow
lns_dict = {'exp': 'solid', 'ctrl': 'dashed'}

ymin = 0; ymax = 30; yint = 10
weight_df_dict = {}
for subplot, (relevant_mice, lbl, tlt_lbl) in enumerate(zip(
        [exp_mice_f, exp_mice_m, ctrl_mice_f, ctrl_mice_m],
        ['MSA-deficient', 'MSA-deficient', 'CTRL littermates', 'CTRL littermates'],
        ['♀', '♂', '♀', '♂']
        )):  
    clr = clr_dict['exp'] if 'MSA' in lbl else clr_dict['ctrl']
    lns = lns_dict['exp'] if 'MSA' in lbl else lns_dict['ctrl']
    subplot_subtract = 1 if '♀' in tlt_lbl else 2
    weight_df_sub = weight_df.loc[relevant_mice, :]
    weight_df_dict[f"{lbl}_{tlt_lbl}"] = weight_df_sub
    sd = np.nanstd(weight_df_sub, axis = 0)
    ax[rows-1, cols-subplot_subtract].fill_between(
                  weight_df.columns, 
                  np.nanmean(weight_df_sub, axis = 0)-sd,
                  np.nanmean(weight_df_sub, axis = 0)+sd,
                  facecolor = clr,
                  alpha = 0.2
                  )
    ax[rows-1, cols-subplot_subtract].plot(
             weight_df.columns,
             np.nanmean(weight_df_sub, axis = 0),
             color = clr,
             linestyle = lns,
             lw = 1.5,
             label = lbl)
    
    if subplot < 2:
        ax[rows-1, cols-subplot_subtract].set_ylim(ymin, ymax)
        ax[rows-1, cols-subplot_subtract].set_yticks(np.arange(ymin,ymax+yint, yint))
        ax[rows-1, cols-subplot_subtract].set_ylabel("Weight (g)")
        ax[rows-1, cols-subplot_subtract].set_title(f"Weight {tlt_lbl}")#, bbox = dict(pad = 0.1, fc = '#ffcc00', ec = 'none'))
        ax[rows-1, cols-subplot_subtract].set_xlabel("Age (days)")
        ax[rows-1, cols-subplot_subtract].set_xlim(20,120)
        ax[rows-1, cols-subplot_subtract].set_xticks([20,70,120])
        
# STATS
pvals, ptexts = get_stats(weight_df_dict['MSA-deficient_♂'], stats_days, second_sample=weight_df_dict['CTRL littermates_♂']) 
for d, p in zip(stats_days, ptexts):
    ax[2,1].text(d, ymax-2, p, fontsize=5, ha='center')
print(f"Male weight:\nDay {stats_days[0]}: {pvals[0]}\nDay {stats_days[1]}: {pvals[1]}\nDay {stats_days[2]}: {pvals[2]}")
   

pvals, ptexts = get_stats(weight_df_dict['MSA-deficient_♀'], stats_days, second_sample=weight_df_dict['CTRL littermates_♀']) 
for d, p in zip(stats_days, ptexts):
    ax[2,2].text(d, ymax-3, p, fontsize=5, ha='center')
print(f"Female weight:\nDay {stats_days[0]}: {pvals[0]}\nDay {stats_days[1]}: {pvals[1]}\nDay {stats_days[2]}: {pvals[2]}")       

ax[rows-1, cols-1].legend(ncol = 1, loc = 'upper left',
            bbox_to_anchor=(0,0.43,0.8,0.2)) 

fig.tight_layout(h_pad = 3)

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], f'FIGURE_bodyscores_weight_{appdx}.svg'),
            dpi=300, transparent=True)