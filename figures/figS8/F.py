from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from scipy.optimize import curve_fit
from scipy.stats import t
def linear(x,A,B):
    return A - B*x

def log_likelihood(ssr,n):
    return -n/2 * (np.log(2 * np.pi * ssr/n)+1)

def plot_figS8F(dep_var, group_num=5):
    fig, ax = plt.subplots(1, 1, figsize=(1.35, 1.4), sharex = True)
    bin_edges = np.linspace(-40.0001,40, group_num+1, endpoint=True)
    
    x_data_comb = np.empty(0)
    y_data_comb = np.empty(0)
    n_data = []
    ssrs = []
    for i, (yyyymmdd, otp_dir, appdx, indep_var, appdx2, clr, lnst, lbl) in enumerate(zip(
                                    ['2022-05-06', '2022-08-18'], # yyyymmdd
                                    ['mtTreadmill', 'passiveOpto'], # otp_dir
                                    ['_egocentric', '_locom_levels_incline'], # appdx
                                    ['trialType', 'trialType'], # indep_var
                                    ['mtTreadmill_levels','locom_levels_incline'], # appdx2
                                    [FigConfig.colour_config['headbars'], FigConfig.colour_config['homolateral'][2]], 
                                    ['dashed', 'solid'],
                                    ['Motorised', 'Passive']
                                    )):
        df = pd.read_csv(os.path.join(Config.paths[f"{otp_dir}_output_folder"], yyyymmdd + f'_limbPositionRegressionArray{appdx}.csv'))
        modLIN = pd.read_csv(os.path.join(Config.paths[f"{otp_dir}_output_folder"], yyyymmdd + f'_limbPositionRegressionArray_MIXEDMODEL_linear_{appdx2}.csv'), index_col=0)
    
        df[indep_var] = [-int(x[3:]) for x in df[indep_var]]
        zero_incline_centred = 0 - np.nanmean(df[indep_var])  # this is zero incline centred
        # compute the Y value for zero incline, later it will be subtracted from 
        # each actual Y value - to centre the Y around zero
        model_Yzero = modLIN.loc[f'Intercept_{dep_var}','Estimate'] + np.nanmean(df[dep_var]) + modLIN.loc[f'param_centred_{dep_var}','Estimate'] * zero_incline_centred
    
        # APPROXIMATE WITH A FUNCTION        
        df[dep_var] = (df[dep_var] - model_Yzero)/Config.passiveOpto_config["px_per_cm"][dep_var[:-1]] # convert to cm
    
        df['indep_bins'], bin_edges = pd.cut(df[indep_var], bins=bin_edges, retbins = True, labels = False)
        # df_sub['indep_bins'] = pd.cut(df_sub[indep_var], bins = bin_edges, labels = False)
        summary = df.groupby('indep_bins')[dep_var].agg(['std', 'sem']).reset_index()
        summary['bin_x'] = [np.mean((bin_edges[i],bin_edges[i+1])) for i in range(bin_edges.shape[0]-1)]
        
        x_pred = np.linspace(summary['bin_x'].min(), summary['bin_x'].max(), endpoint=True)
        mask = ~np.isnan(df[dep_var].values)
        popt,pcov = curve_fit(linear, df[indep_var].values[mask], df[dep_var].values[mask], p0=(np.nanmean(df[dep_var].values),
                                                                                      (np.nanmax(df[dep_var].values)-np.nanmin(df[dep_var].values))/(np.nanmax(df[indep_var].values)-np.nanmin(df[indep_var].values)),
                                                                                      ))
        A_fit, B_fit = popt
        print(f"Linear fitted params: A = {A_fit:.3f}, B = {B_fit:.3f}")
        y_pred = linear(x_pred, *popt)
        
        # 95% confidence intervals
        ids = np.linspace(0, len(x_pred)-1, group_num, dtype=int)
        summary['ci95_hi'] = y_pred[ids] + summary['std'] + 1.96*summary['sem']
        summary['ci95_lo'] = y_pred[ids] - summary['std'] - 1.96*summary['sem']
        ax.fill_between(
            x_pred[ids],
            summary['ci95_lo'],
            summary['ci95_hi'],
            alpha = 0.3,
            color = clr,
            edgecolor = None,
            )
        
        ax.plot(x_pred,#+(i*0.05), 
                      y_pred, 
                      linewidth=1.5, 
                      linestyle = lnst,
                      color=clr,
                      label = lbl)
        
        ax.set_xlim(-50,50)
        ax.set_xticks([-40,0,40], labels = [-40,0,40])
        ax.set_ylim(-2.5,3)
        ax.set_yticks(np.arange(-2,4))
        ax.set_title("Slope trials")
        
        ## plot horizontal line as a legend
        ax.hlines(2.85, 2- (i*44), 31 - (i*40), color = clr, ls = lnst, lw = 1)
        
        # COMBINE DATA
        x_data_comb = np.concatenate((x_data_comb, df[indep_var].values[mask]))
        y_data_comb = np.concatenate((y_data_comb, df[dep_var].values[mask]))
        n_data.append(df[indep_var].values[mask].shape[0])
        ssrs.append(np.sum((df[dep_var].values[mask] - linear(df[indep_var].values[mask], *popt))**2))
        
        # COMPUTE INDIVIDUAL P-VALUES
        std_err = np.sqrt(np.diag(pcov)) # standard deviations in 
        t_values = popt/std_err
        dof = max(0, len(df[dep_var].values)-len(popt))   
        p_values = [2 * (1 - t.cdf(np.abs(t_val), dof)) for t_val in t_values]
        print(f"{otp_dir} p-values: A_p = {p_values[0]:.3e}, B_p = {p_values[1]:.3e}")
        
    plt.tight_layout()
    
    # COMPUTE F STATISTIC
    n_data1, n_data2 = n_data
    ssr1, ssr2 = ssrs
    logL1 = log_likelihood(ssr1, n_data1)
    logL2 = log_likelihood(ssr2, n_data2)
    
    from scipy.stats import chi2
    logL_comb = log_likelihood(ssr1+ssr2, n_data1+n_data2)
    lr_stat = -2 * (logL_comb - (logL1+logL2))
    dof = len(popt)
    p_value = chi2.sf(lr_stat,dof)
    print(f"Comparison p-value: {p_value}")
    
    ax.text(-44,3, "passive ", ha = 'left', 
            color = FigConfig.colour_config['homolateral'][2], 
            fontsize = 5)
    p_text = "vs motorised: " + ('*' * (p_value < FigConfig.p_thresholds).sum())
    if (p_value < FigConfig.p_thresholds).sum() == 0:
        p_text += "n.s."
    ax.text(-12,3, p_text, ha = 'left', 
            color = FigConfig.colour_config['headbars'], 
            fontsize = 5)
    
    
    ax.text(-50.5,3.3,'anterior', ha = 'right', fontsize = 5)
    ax.text(-50.5,-3,'posterior', ha = 'right', fontsize = 5)
    
    ax.set_xlabel('Surface slope\n(deg)')
    ax.set_ylabel('Horizontal position of\nRH foot (cm)')
    plt.tight_layout()
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    savepath = Path(FigConfig.paths['savefig_folder']) / "forceplate_foorPosition_vs_slope_COMPARISON.svg"
    fig.savefig(savepath,
                dpi=300,
                transparent=True)
    print(f"FIGURE SAVED AT {savepath}")
if __name__=="__main__":   
    dep_var = 'rH1x'
    plot_figS8F(dep_var)