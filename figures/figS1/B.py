import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def plot_figS1B(yyyymmdd, param):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="meanParamDF", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = f'_{param}')
    
    mice = np.unique(df['mouse'])
    fig, axes = plt.subplots(1, 1, figsize=(1.3, 1.47))
    
    limb_clr = 'homolateral'
    limb_str = 'hind_weight_frac'
    tlt = 'Head height trials'
    variable_str = 'hindWfrac'
    perc_changes = []
    for im, m in enumerate(mice):
        df_sub = df[df['mouse'] == m]
        headHWs = np.unique(df_sub['param'])
        
        yvals = [np.nanmean(df_sub[df_sub['param']==h][limb_str]) for h in np.unique(df_sub['param'])]
        axes.plot(headHWs, 
                 yvals, 
                 color=FigConfig.colour_config[limb_clr][2],  
                 alpha=0.4, 
                 linewidth = 0.7)
        perc_changes.append((yvals[-1]-yvals[0])*100/yvals[0])        
    # # fore-hind and comxy plot means
    
    # LOAD MIXED-EFFECTS MODEL
    slope_enforced = 'slopeENFORCED'
    mod = 'Slope'
    path = f"{Config.paths['forceplate_output_folder']}\\{yyyymmdd}_mixedEffectsModel_linear_{variable_str}_{param}_{slope_enforced}_rand{mod}.csv"
    stats_df = pd.read_csv(path, index_col=0)
    
    # A + Bx
    x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)
    x_centred = x_pred - np.nanmean(df['param'].values)
    y_centred = stats_df.loc['(Intercept)', 'Estimate'] + (stats_df.loc['indep_var_centred', 'Estimate'] * x_centred)
    y_pred = y_centred + np.nanmean(df[limb_str].values)
    
    axes.plot(x_pred, 
                  y_pred, 
                  linewidth=1.5, 
                  color=FigConfig.colour_config[limb_clr][2])
    
    # PLOT STATS
    t = stats_df.loc['indep_var_centred', 't value']
    p = stats_df.loc['indep_var_centred', 'Pr(>|t|)']
    print(f"{limb_str}: mean={stats_df.loc['indep_var_centred', 'Estimate']:.4g}, SEM={stats_df.loc['indep_var_centred', 'Std. Error']:.4g}, t({stats_df.loc['indep_var_centred', 'df']:.0f})={t:.3f}, p={p:.3g}")
    p_text = "n.s." if (p < FigConfig.p_thresholds).sum() == 0 else ('*' * (p < FigConfig.p_thresholds).sum())
    
    axes.text(0.6,
                  1.46, 
                  f"slope: {p_text}", 
                  ha = 'center', 
                  color = FigConfig.colour_config[limb_clr][2],
                  fontsize = 5)
    
    print(f"Hindlimb load changed by {np.mean(perc_changes):.3g} ± {np.std(perc_changes)/np.sqrt(len(perc_changes)):.3g}%")
    
    # if A is significant, it means that the data asymptotes at a non-zero value
    # if B is significant, it means that there is a (monotonic?) change in y as the x changes
    # if k is significant, it means that y approaches the asymptote in an exponential manner
         
    axes.set_xlabel('Weight-adjusted\nhead height')
    
    axes.set_xticks([0,0.6,1.2])
    axes.set_xlim(0,1.2)
    axes.set_yticks([0,0.5,1.0,1.5])
    axes.set_ylim(-0.1,1.5)
    axes.set_title(tlt)
    
    axes.set_ylabel("Hindlimb weight fraction")
    plt.tight_layout()
    
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    savepath = os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_hindHeadWeights_{param}_{limb_str}.svg')
    fig.savefig(savepath,
                transparent = True,
                dpi = 300)
    print(f"FIGURE SAVED AT {savepath}")

if __name__=="__main__":
    yyyymmdd = '2021-10-26'
    param = 'headHW'
    plot_figS1B(yyyymmdd, param)