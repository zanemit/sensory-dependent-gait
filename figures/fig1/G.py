import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def plot_fig1G(yyyymmdd, param):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="meanParamDF", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = f'_{param}')
    
    mice = np.unique(df['mouse'])
    fig, ax = plt.subplots(1, 1, figsize=(1.34, 1.47))
    ax.hlines(100,xmin=-40,xmax=40, ls = 'dashed', color = 'grey')
    
    clr = 'greys'
    variable_str = 'headplate_weight_frac'
    variable = 'headWfrac'
    
    # CONVERT HEADPLATE VALUES TO PERCENTAGE OF DETECTED WEIGHT
    df[variable_str] = -(df[variable_str]-1)*100
    
    starts = np.empty(len(mice)); ends = np.empty(len(mice))
    for im, m in enumerate(mice):
        df_sub = df[df['mouse'] == m]
        headHWs = np.unique(df_sub['param'])
    
        yvals = [np.nanmean(df_sub[df_sub['param']==h][variable_str]) for h in np.unique(df_sub['param'])]
        starts[im] = yvals[0]; ends[im] = yvals[-1]
        ax.plot(headHWs, 
                 yvals, 
                 color=FigConfig.colour_config[clr][1],  
                 alpha=0.4, 
                 linewidth = 0.7)
    
    print(f"Max decline: {np.mean(starts):.3f} ± {scipy.stats.sem(starts):.3f}")
    print(f"Max incline: {np.mean(ends):.3f} ± {scipy.stats.sem(ends):.3f}")
    
    # LOAD MIXED-EFFECTS MODEL
    slope_enforced = 'slopeENFORCED'
    mod = 'Slope'
    path = f"{Config.paths['forceplate_output_folder']}\\{yyyymmdd}_mixedEffectsModel_linear_{variable}_{param}_{slope_enforced}_rand{mod}.csv"
    stats_df = pd.read_csv(path, index_col=0)
    
    # A + Bx
    x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)
    x_centred = x_pred - np.nanmean(df['param'].values)
    y_centred = stats_df.loc['(Intercept)', 'Estimate'] + (stats_df.loc['indep_var_centred', 'Estimate'] * x_centred)
    y_pred = y_centred + np.nanmean(df[variable_str].values)
    
    ax.plot(x_pred, 
            y_pred, 
            linewidth=1.5, 
            color=FigConfig.colour_config[clr][1])
    
    # PLOT STATS
    t = stats_df.loc['indep_var_centred', 't value']
    p = stats_df.loc['indep_var_centred', 'Pr(>|t|)']
    print(f"{variable_str}: mean={stats_df.loc['indep_var_centred', 'Estimate']:.4g}, SEM={stats_df.loc['indep_var_centred', 'Std. Error']:.4g}, t({stats_df.loc['indep_var_centred', 'df']:.0f})={t:.3f}, p={p:.3g}")
    p_text = "n.s." if (p < FigConfig.p_thresholds).sum() == 0 else ('*' * (p < FigConfig.p_thresholds).sum())
    
    ax.text(0,175, f"slope: {p_text}", ha = 'center', 
            color = FigConfig.colour_config[clr][1], 
            fontsize = 5)
            
    ax.set_xlabel('Surface slope\n(deg)')
    
    ax.set_xticks([-40,-20,0,20,40][::2])
    ax.set_xlim(-50,50)
    ax.set_yticks([60,100,140,180])
    ax.set_ylim(40,180)
    ax.set_title("Surface slope trials")
    
    
    ax.set_ylabel("Total leg load (%)")
    plt.tight_layout()
    
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    savepath = os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_totalWeight_{param}.svg')
    fig.savefig(savepath,
            transparent = True,
            dpi = 300)
    print(f"FIGURE SAVED AT {savepath}")

if __name__=="__main__":
    yyyymmdd = '2022-04-04'
    param = 'levels'
    plot_fig1G(yyyymmdd, param)