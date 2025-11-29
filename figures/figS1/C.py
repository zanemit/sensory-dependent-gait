import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def plot_figS1C(yyyymmdd, param):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="meanParamDF", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = f'_{param}')
    
    mice = np.unique(df['mouse'])
    fig, ax = plt.subplots(1, 1, figsize=(1.33, 1.47))
    ax.hlines(100,xmin=0,xmax=1.2, ls = 'dashed', color = 'grey')
    
    limb_clr = 'greys'
    limb_str = 'headplate_weight_frac'
    
    # CONVERT HEADPLATE VALUES TO PERCENTAGE OF DETECTED WEIGHT
    df[limb_str] = -(df[limb_str]-1)*100
    
    for im, m in enumerate(mice):
        df_sub = df[df['mouse'] == m]
        headHWs = np.unique(df_sub['param'])
        
        yvals = [np.nanmean(df_sub[df_sub['param']==h][limb_str]) for h in np.unique(df_sub['param'])]
        ax.plot(headHWs, 
                 yvals, 
                 color=FigConfig.colour_config[limb_clr][1],  
                 alpha=0.4, 
                 linewidth = 0.7)
    
    
    # APPROXIMATE WITH A FUNCTION
    from scipy.optimize import curve_fit
    from scipy.stats import wilcoxon
    def exp_decay(x,A,B,k):
        return A - B * np.exp(-k*x)
    
    x_pred = np.linspace(np.nanmin(df['param'].values), np.nanmax(df['param'].values), endpoint=True)
    
    # TOTAL FOR PLOTTING
    popt,pcov = curve_fit(exp_decay, df['param'].values, df[limb_str].values, p0=(np.nanmax(df[limb_str].values),
                                                                                  np.nanmax(df[limb_str].values)-np.nanmin(df[limb_str].values),
                                                                                  1/np.nanmean(df['param'].values)))
    A_fit, B_fit, k_fit = popt
    print(f"Exp decay fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}, k = {k_fit:.3f}")
    ax.plot(x_pred, 
                  exp_decay(x_pred, *popt), 
                  linewidth=1.5, 
                  color=FigConfig.colour_config[limb_clr][1])
    
    # STATS PER SUBJECT
    p_list = []
    for m in df['mouse'].unique():
        df_sub = df[df['mouse']==m].copy()
        popt, pcov = curve_fit(exp_decay, df_sub['param'].values, df_sub[limb_str].values, p0=(A_fit, B_fit, k_fit), maxfev=10000)
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
    
        ax.text(0.6,
                  160-(i_p*10), 
                  f"{exp_d_param}: {p_text}", 
                  ha = 'center', 
                  color = FigConfig.colour_config[limb_clr][0],
                  fontsize = 5)
    # if A is significant, it means that the data asymptotes at a non-zero value
    # if B is significant, it means that there is a (monotonic?) change in y as the x changes
    # if k is significant, it means that y approaches the asymptote in an exponential manner
    
            
    ax.set_xlabel('Weight-adjusted\nhead height')
    
    ax.set_xticks([0,0.6,1.2])
    ax.set_xlim(-0.1,1.2)
    ax.set_yticks([40,70,100,130,160])
    ax.set_ylim(30,160)
    ax.set_title("Head height trials")
    
    
    ax.set_ylabel("Total leg load (%)")
    plt.tight_layout()
    
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    savepath = os.path.join(FigConfig.paths['savefig_folder'], f'forceplate_totalWeight_{param}_{limb_str}.svg')
    fig.savefig(savepath,
            transparent = True,
            dpi = 300)
    print(f"FIGURE SAVED AT {savepath}")
    
if __name__=="__main__":
    yyyymmdd = '2021-10-26'
    param = 'headHW'
    plot_figS1C(yyyymmdd, param)
