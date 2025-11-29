from pathlib import Path
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import stats
from processing import data_loader, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def compute_forelimb_phase(ref_phases, nonref_phases):
    phases = nonref_phases-ref_phases
    phases[phases<-0.5] = phases[phases<-0.5]+1
    phases[phases>0.5] = phases[phases>0.5]-1
    return phases

def plot_figS2L(yyyymmdd, param, refLimb, appdx=""):
    outputDir = Config.paths["mtTreadmill_output_folder"]

    appdx = ""
    kdeplot = False
    kde_bin_num = 100
        
    if refLimb == 'lH1':
        limblist = {'homologous': 'rH0', 'homolateral': 'lF0', 'diagonal': 'rF0'}
        side_str = 'right'
    elif refLimb == 'rH1':
        limblist = {'homologous': 'lH0', 'homolateral': 'rF0', 'diagonal': 'lF0'}
        side_str = 'left'
    
    
    df, _, _ = data_loader.load_processed_data(outputDir, 
                                    dataToLoad = 'strideParams', 
                                    appdx = appdx,
                                    yyyymmdd = yyyymmdd,
                                    limb = refLimb)
    
    mouselist = Config.mtTreadmill_config['egr3_ctrl_mice']
    
    # df = df[df['speed'] > np.nanpercentile(df['speed'],50) ]      
    df['hind_phase'] = df[limblist['homologous']]
    df['fore_phase'] = compute_forelimb_phase(ref_phases = df[limblist['homolateral']], nonref_phases = df[limblist['diagonal']])
    limbs = ['hind', 'fore']
    limb_str = ['homologous', 'diagonal']
    
    upper_lim = 0.4; lower_lim = 0.1  
    
    ########## PLOT BOXPLOTS ########## 
    fig, ax = plt.subplots(1,1, figsize = (1.4,1.3))
    injection_preference = {}
    for s, inj_side in enumerate(['right', 'left']):
        mice = np.intersect1d(mouselist, Config.injection_config[f'{inj_side}_inj_imp'])
        
        fracKDE_R_after_L_across_mice = np.empty((len(mice), len(limbs))) *np.nan
        frac_R_after_L_across_mice = np.empty((len(mice), len(limbs))) * np.nan
        
        for il, limb in enumerate(limbs):
            # restrict data to out-of-phase coordination
            phase_str = f'{limb}_phase'
            df_sub = df.iloc[np.where((df[phase_str]<=upper_lim) & (df[phase_str]>=-upper_lim))[0], :]
            df_sub = df_sub.iloc[np.where((df_sub[phase_str]>=lower_lim) | (df_sub[phase_str]<=-lower_lim))[0], :]
            
            for im, m in enumerate(mice):
                df_sub_m = df_sub[df_sub['mouseID'] == m]
                values = df_sub_m[phase_str]
                
                # relevant if kde=True
                kde_bins, kde = utils_math.von_mises_kde_exact(values*2*np.pi, 10, kde_bin_num)
                kde_bins_points = np.concatenate((kde_bins, [kde_bins[0]]))
                kde_points = np.concatenate((kde, [kde[0]]))
                   
                # right after left
                kde_points_suprazero = kde_points[:-1][kde_bins_points[:-1]>0]
                kde_bins_points_suprazero = kde_bins_points[:-1][kde_bins_points[:-1]>0]
                area_suprazero = np.trapz(kde_points_suprazero, kde_bins_points_suprazero)
                
                # left after right
                kde_points_subzero = kde_points[:-1][kde_bins_points[:-1]<0]
                kde_bins_points_subzero = kde_bins_points[:-1][kde_bins_points[:-1]<0]
                area_subzero = np.abs(np.trapz(kde_points_subzero, kde_bins_points_subzero))
                
                area_total = area_suprazero + area_subzero
                
                fracKDE_R_after_L_across_mice[im, il] = area_suprazero/area_total
                
                # compute the same but as regular fractions! - relevant if kde = False
                # right after left
                points_suprazero = (values>0).sum()
                points_subzero = (values<0).sum()
                frac_R_after_L_across_mice[im, il] = points_suprazero / (points_suprazero + points_subzero)
            
            if kdeplot:
                frac_arr = fracKDE_R_after_L_across_mice
            else:
                frac_arr = frac_R_after_L_across_mice
            
            scatter_shift = 0.2-(il*0.4)
            boxplot_shift = -0.1+(il*0.2)
            
            clr = FigConfig.colour_config[limb_str[il]][2]
            ax.boxplot(frac_arr[:, il], positions = [il+(2*s)+boxplot_shift], 
                       medianprops = dict(color = clr, linewidth = 1, alpha = 0.6),
                       boxprops = dict(color = clr, linewidth = 1, alpha = 0.6), 
                       capprops = dict(color = clr, linewidth = 1, alpha = 0.6),
                       whiskerprops = dict(color = clr, linewidth = 1, alpha = 0.6), 
                       flierprops = dict(mec = clr, linewidth = 1, alpha = 0.6, ms=2))
            ax.scatter(np.repeat(il+(2*s)+scatter_shift, 
                       frac_arr.shape[0]), 
                       frac_arr[:,il], 
                       color =  clr, alpha = 0.8, s = 5, zorder = 3)
            
            # store frac_arr for stats later
            injection_preference[inj_side] = frac_arr
        
        for i in range(frac_arr.shape[0]):
            clr = [FigConfig.colour_config[limb_str[0]][2], FigConfig.colour_config[limb_str[1]][2]]
            ax.plot([0+(s*2)-scatter_shift,1+(s*2)+scatter_shift], frac_arr[i, :], linewidth = 1, color = '#b17242', alpha = 0.2)
    ax.axhline(0.5, 0.1, 0.9, ls = 'dashed', color = 'grey')
    
    # ###### STATS ###########
    limbs = ['hind', 'fore']
    for inj_id, (hline_X1, hline_X2, hline_Y, text_Y) in enumerate(zip(
            [0.1, 1], [2.1,3], [0.25,0.77], [0.17,0.80]
            )):
        _, pval = stats.ttest_ind(injection_preference['right'][:,inj_id], injection_preference['left'][:,inj_id])
        
        p_text = ('*' * (pval < (np.asarray(FigConfig.p_thresholds))).sum())
        if (pval < (np.asarray(FigConfig.p_thresholds)).sum()) == 0 and not np.isnan(pval):
            p_text = "n.s."
            
        ax.hlines(hline_Y, hline_X1, hline_X2, color='grey')
        ax.text(inj_id+1, text_Y, p_text, ha='center', fontsize=5, color='grey')
        print(f"{limbs[inj_id]}: {pval:.3g}")
        
    ax.set_ylim(0,1)
    ax.set_yticks(np.linspace(0,1, 3, endpoint=True))
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(limbs*2, fontsize=5.75)
    ax.text(0.5,0.97,"RIGHT", ha='center', color = 'grey', fontsize=5)
    ax.text(2.5,0.97,"LEFT", ha='center', color = 'grey', fontsize=5)
    ax.set_ylabel(f"Fraction of {side_str}-leading\nout-of-phase strides")
    ax.set_title("Motorised treadmill")
    fig.tight_layout()
    
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    filename = "mtTreadmill_lateralisation.svg"
    savepath = Path(FigConfig.paths['savefig_folder']) / filename
    fig.savefig(savepath, dpi = 300, transparent=True)  
    print(f"FIGURE SAVED AT {savepath}")
    
if __name__=="__main__":
    param = 'homologous_gait'
    yyyymmdd = '2023-09-25'
    refLimb = 'lH1'
    plot_figS2L(yyyymmdd, param, refLimb)
