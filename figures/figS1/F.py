from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from processing import data_loader
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def plot_figS1F(yyyymmdd):
    df, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                   dataToLoad="forceplateAngleParams", 
                                                   yyyymmdd = yyyymmdd, 
                                                   appdx = '_headHW')
    modQDRhw = pd.read_csv(Path(Config.paths["forceplate_output_folder"])/f"{yyyymmdd}_mixedEffectsModel_quadratic_snoutBodyAngle_headHW.csv")
    
    mice = np.unique(df['mouseID'])
    
    fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.47))
    for m in mice:
        df_sub = df[df['mouseID'] == m]
        headHWs = np.unique(df_sub['headHW'])
        bodyAngles = [np.nanmean(df_sub[df_sub['headHW']==h]['snoutBodyAngle']) for h in np.unique(df_sub['headHW'])]
        ax.plot(headHWs, 
                bodyAngles, 
                color = FigConfig.colour_config['headbars'], 
                alpha=0.4, 
                linewidth = 0.7)
    
    # APPROXIMATE WITH A FUNCTION
    from scipy.optimize import curve_fit
    from scipy.stats import wilcoxon
    def exp_decay(x,A,B,k):
        return A - B * np.exp(-k*x)
    
    x_pred = np.linspace(np.nanmin(df['headHW'].values), np.nanmax(df['headHW'].values), endpoint=True)
    
    # TOTAL FOR PLOTTING    
    mask = ~np.isnan(df['snoutBodyAngle'].values)
    popt,pcov = curve_fit(exp_decay, 
                          df['headHW'].values[mask], 
                          df['snoutBodyAngle'].values[mask], 
                          p0=(
                               np.nanmax(df['snoutBodyAngle'].values),
                            np.nanmax(df['snoutBodyAngle'].values)-np.nanmin(df['snoutBodyAngle'].values),
                            1/np.nanmean(df['headHW'].values))
                          )
    A_fit, B_fit, k_fit = popt
    ax.plot(x_pred, 
                  exp_decay(x_pred, *popt), 
                  linewidth=1.5, 
                  color=FigConfig.colour_config['headbars'])
    
    # STATS PER SUBJECT
    p_list = []
    for m in mice:
        df_sub = df[df['mouseID']==m].copy()
        mask = ~np.isnan(df_sub['snoutBodyAngle'].values)
        popt, pcov = curve_fit(exp_decay, df_sub['headHW'].values[mask], 
                                df_sub['snoutBodyAngle'].values[mask], 
                        p0=(
                            A_fit,
                            B_fit,
                            k_fit
                            ), 
                        maxfev=10000)
        p_list.append(popt)
    
        
    params = np.array(p_list) #(n_subjects, n_params)
    param_names = ['A', 'B', 'k']
    params_df = pd.DataFrame(params, columns=param_names, index=df['mouseID'].unique())
    
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
                  179-(i_p*3.5), 
                  f"{exp_d_param}: {p_text}", 
                  ha = 'center', 
                  color = FigConfig.colour_config['headbars'],
                  fontsize = 5)
    
    ax.set_xlim(0,1.2)
    ax.set_ylim(140,180)
    ax.set_ylabel('Snout-hump angle\n(deg)')
    ax.set_xlabel('Weight-adjusted\nhead height')
    ax.set_xticks([0,0.4,0.8,1.2])
    ax.set_yticks([140,150,160,170,180])
    ax.set_title("Head height trials")
    
    plt.tight_layout()
    
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    savepath= Path(FigConfig.paths['savefig_folder']) / "forceplate_snoutBodyAngles_vs_headHW.svg"
    fig.savefig(savepath,
                transparent = True,
                dpi=300)
    print(f"FIGURE SAVED AT {savepath}")
    
if __name__=="__main__":
    yyyymmdd = '2021-10-26'    
    plot_figS1F(yyyymmdd)
    
