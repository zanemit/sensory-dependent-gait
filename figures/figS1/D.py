import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def plot_figS1D(yyyymmdd, param):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="meanParamDF", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = '_'+param)
    
    mice = np.unique(df['mouse'])
    
    fig, axes = plt.subplots(1,1, figsize=(1.45, 1.47))
    
    variable = 'CoMy_mean' #'CoMx_mean'
    clr = 'homolateral' # 'greys'
    variable_str ='CoMy' # 'CoMx'
        
    diffs = np.empty(len(mice))
    stds = np.empty((len(mice), 8))*np.nan
    for im, m in enumerate(mice):
        df_sub = df[df['mouse'] == m]
        headHWs = np.unique(df_sub['param'])
        
        yvals = [np.nanmean(df_sub[df_sub['param']==h][variable]*Config.forceplate_config['fore_hind_post_cm']/2) for h in np.unique(df_sub['param'])]
        
        # quantify variability
        stds[im, :len(headHWs)] = [np.nanstd(df_sub[df_sub['param']==h][variable]*Config.forceplate_config['fore_hind_post_cm']/2) for h in np.unique(df_sub['param'])]
        
        diffs[im] = yvals[-1]-yvals[0]
        axes.plot(headHWs, 
                 yvals, 
                 color=FigConfig.colour_config[clr][2],  
                 alpha=0.4, 
                 linewidth = 0.7)
    print(f"Average standard deviation: {np.mean(np.nanmean(stds, axis=1))}")
    print(f"Mean change over 80 degrees: {np.mean(diffs):.3f} ± {scipy.stats.sem(diffs):.3f}")
    
    x_centered = df['param'] - np.nanmean(df['param'])
    x_pred = np.linspace(np.nanmin(x_centered), np.nanmax(x_centered), endpoint=True)
    
    
    # APPROXIMATE WITH A FUNCTION
    from scipy.optimize import curve_fit
    from scipy.stats import wilcoxon
    def exp_decay(x,A,B,k):
        return A - B * np.exp(-k*x)
    
    x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)
    #  TOTAL FOR PLOTTING
    popt,pcov = curve_fit(exp_decay, df['param'].values, df[variable].values*Config.forceplate_config['fore_hind_post_cm']/2, p0=(np.nanmax(df[variable].values)*Config.forceplate_config['fore_hind_post_cm']/2,
                                                                                  (np.nanmax(df[variable].values)-np.nanmin(df[variable].values))*Config.forceplate_config['fore_hind_post_cm']/2,
                                                                                  1/np.nanmean(df['param'].values)))
    A_fit, B_fit, k_fit = popt
    print(f"Exp decay fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}, k = {k_fit:.3f}")
    axes.plot(x_pred, 
                  exp_decay(x_pred, *popt), 
                  linewidth=1.5, 
                  color=FigConfig.colour_config[clr][2])
    
    # STATS PER SUBJECT
    p_list = []
    for m in df['mouse'].unique():
        df_sub = df[df['mouse']==m].copy()
        popt, pcov = curve_fit(exp_decay, df_sub['param'].values, df_sub[variable].values, p0=(A_fit, B_fit, k_fit), maxfev=10000)
        p_list.append(popt)
        
    params = np.array(p_list) #(n_subjects, n_params)
    param_names = ['A', 'B', 'k']
    params_df = pd.DataFrame(params, columns=param_names, index=df['mouse'].unique())
    
    # STATS ACROSS SUBJECTS
    summary = []
    for col in param_names:
        vals = params_df[col].dropna().values
        median = np.median(vals)
        w_stat, p = wilcoxon(vals)
        summary.append((col, median, w_stat, p))
    for col, median,  w_stat, p in summary:
        print(f"{col}: median={median:.4g}, w({params_df.shape[0]-1})={w_stat:.3f}, p={p:.3g}")
    
    # PLOT STATS
    for i_p, (p_k, exp_d_param) in enumerate(zip(
            [summary[1][3], summary[2][3]],
            ['scale factor', 'rate constant']
            )):
        p_text = "n.s." if (p_k < FigConfig.p_thresholds).sum() == 0 else ('*' * (p_k < FigConfig.p_thresholds).sum())
    
        axes.text(0.63,
                  0.9-(i_p*0.2), 
                  f"{exp_d_param}: {p_text}", 
                  ha = 'center', 
                  color = FigConfig.colour_config['homolateral'][2],
                  fontsize = 5)
    
    axes.set_xlabel('Weight-adjusted\nhead height')
    axes.set_xticks([0,0.6,1.2])
    axes.set_xlim(0,1.2)
    axes.set_yticks([-1.5,-1.0,-0.5,0,0.5,1.0])
    axes.set_ylim(-1.5,1.0)
    axes.set_title("Head height trials")
    
    axes.set_ylabel("Anteroposterior\nCoS (cm)")
    plt.tight_layout()
    
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    savepath = os.path.join(FigConfig.paths['savefig_folder'], 'forceplate_CoMy_vs_headHW.svg')
    fig.savefig(savepath,
                transparent = True,
                dpi =300)
    print(f"FIGURE SAVED AT {savepath}")
    
if __name__=="__main__":
    yyyymmdd = '2021-10-26'
    param = 'headHW'
    plot_figS1D(yyyymmdd, param)
