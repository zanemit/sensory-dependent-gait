import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def plot_fig1F(yyyymmdd, param):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="meanParamDF", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = '_'+param)
    
    mice = np.unique(df['mouse'])
    
    fig, ax = plt.subplots(1,1, figsize=(1.45, 1.47))
    
    variable = 'CoMy_mean'
    clr = 'homolateral'
    variable_str = 'CoMy'
    tlt = 'Surface slope trials'
    
    stds1 = np.empty((len(mice), 8))*np.nan
    
    diffs = np.empty(len(mice))
    for im, m in enumerate(mice):
        df_sub = df[df['mouse'] == m]
        headHWs = np.unique(df_sub['param'])
        
        yvals = [np.nanmean(df_sub[df_sub['param']==h][variable]*Config.forceplate_config['fore_hind_post_cm']/2) for h in np.unique(df_sub['param'])]
        # quantify variability
        stds1[im, :len(headHWs)] = [np.nanstd(df_sub[df_sub['param']==h][variable]*Config.forceplate_config['fore_hind_post_cm']/2) for h in np.unique(df_sub['param'])]
        
        
        diffs[im] = yvals[-1]-yvals[0]
        ax.plot(headHWs, 
                 yvals, 
                 color=FigConfig.colour_config[clr][2],  
                 alpha=0.4, 
                 linewidth = 0.7)
        
    print(f"Average standard deviation: {np.mean(np.nanmean(stds1, axis=1)):.2f}")
    
    # LOAD MIXED-EFFECTS MODEL
    slope_enforced = 'slopeENFORCED'
    mod = 'Slope'
    path = f"{Config.paths['forceplate_output_folder']}\\{yyyymmdd}_mixedEffectsModel_linear_{variable_str}_{param}_{slope_enforced}_rand{mod}.csv"
    stats_df = pd.read_csv(path, index_col=0)
    
    # A + Bx
    x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)
    x_centred = x_pred - np.nanmean(df['param'].values)
    y_centred = stats_df.loc['(Intercept)', 'Estimate'] + (stats_df.loc['indep_var_centred', 'Estimate'] * x_centred)
    y_pred = (y_centred + np.nanmean(df[variable].values))*Config.forceplate_config['fore_hind_post_cm']/2
    
    ax.plot(x_pred, 
            y_pred, 
            linewidth=1.5, 
            color=FigConfig.colour_config[clr][2])
    
    # PLOT STATS
    t = stats_df.loc['indep_var_centred', 't value']
    p = stats_df.loc['indep_var_centred', 'Pr(>|t|)']
    print(f"{variable}: mean={stats_df.loc['indep_var_centred', 'Estimate']:.4g}, SEM={stats_df.loc['indep_var_centred', 'Std. Error']:.4g}, t({stats_df.loc['indep_var_centred', 'df']:.0f})={t:.3f}, p={p:.3g}")
    p_text = "n.s." if (p < FigConfig.p_thresholds).sum() == 0 else ('*' * (p < FigConfig.p_thresholds).sum())
    
    ax.text(0,0.9, f"slope: {p_text}", ha = 'center', 
            color = FigConfig.colour_config[clr][2], 
            fontsize = 5)
    
    ax.set_xlabel('Surface slope\n(deg)')
    ax.set_xticks([-40,-20,0,20,40][::2])
    ax.set_xlim(-50,50)
    ax.set_yticks([-1.0,-0.5,0,0.5,1])
    ax.set_ylim(-1.3,1)
    ax.set_title(tlt)
    
    ax.set_ylabel("Anteroposterior\nCoS (cm)")
    plt.tight_layout()

    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    savepath = os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_CoMy_vs_incline.svg')
    fig.savefig(savepath,
                transparent = True,
                dpi =300)
    print(f"FIGURE SAVED AT {savepath}")

if __name__=="__main__":
    yyyymmdd = '2022-04-04'
    param = 'levels'
    plot_fig1F(yyyymmdd, param)