import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import os
from processing import utils_math, treadmill_circGLM
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def plot_figS8B():
    predictorlist = ['speed', 'snoutBodyAngle']
    predictorlist_str = ['speed', 'snout-hump angle']
    predictor = 'snoutBodyAngle'
    predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
    appdx =  '' 
    tlt = 'Level trials'
    yyyymmdd = '2021-10-23'
    slopes = ['pred2']
    limb = 'lF0'
    ref = 'lH1LleadRleadalt'
    interaction = 'TRUEthreeway'
    samples = 15159
    datafrac = 0.4
    iters = 1000
    categ_var='rH0_categorical'
    sba_str = 's'
    
    x_range, phase_preds = treadmill_circGLM.get_circGLM_slopes(
            predictors = predictorlist,
            yyyymmdd = yyyymmdd,
            limb = limb,
            ref = ref,
            samples = samples,
            interaction = interaction,
            appdx = appdx,
            datafrac = datafrac,
            categ_var=categ_var,
            slopes = slopes,
            outputDir = Config.paths['mtTreadmill_output_folder'],
            iterations = iters,
            sBA_split_str = sba_str,
            mice = Config.mtTreadmill_config['mice_level']
                    ) 
    
    unique_traces = np.empty((0))
    
    ### PLOTTING
    ylim = (0.3*np.pi,1.5*np.pi)
    yticks = [0.5*np.pi,np.pi,1.5*np.pi]
    yticklabels = ["0.5π", "π", "1.5π"]  
    xlim, xticks, xlabel = treadmill_circGLM.get_predictor_range(predictor)
    xlabel='Snout-hump angle\n(deg)'
    
    fig, ax = plt.subplots(1,1,figsize = (1.35,1.4)) #1.6,1.4 for 4figs S2 bottom row
    
    last_vals = [] # for stats
    
    # plot each mouse (just default ref limb)
    clr = 'homolateral'
    for ref_id, (lnst, lbl) in enumerate(zip(['solid','dotted', 'dashdot'],
                                                ['alt','L-lead','R-lead'])):
        
        c = FigConfig.colour_config[clr][ref_id]
        pp = phase_preds[:, :, predictor_id, 0, ref_id]
        # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
        for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
            print(f"Working on data range {k}...")
            if k == 1:
                pp[pp<0] = pp[pp<0]+2*np.pi
            if k == 2:
                pp[pp<np.pi] = pp[pp<np.pi]+2*np.pi
                ax.hlines(ylim[1]-0.82-0.5*ref_id, 170, 185, color = c, ls = lnst, lw = 0.7)
                ax.text(xlim[0] + (0.8 * (xlim[1]-xlim[0])),
                        ylim[1] - (0.2* (ylim[1]-ylim[0]))- 0.5*ref_id,
                        lbl,
                        color=c,
                        fontsize=5)
                
            trace = scipy.stats.circmean(pp[:,:],high = hi, low = lo, axis = 0)
            lower = np.zeros_like(trace); higher = np.zeros_like(trace)
            for x in range(lower.shape[0]):
                lower[x], higher[x] =  utils_math.hpd_circular(pp[:,x], 
                                                                mass_frac = 0.95, 
                                                                high = hi, 
                                                                low = lo) #% (np.pi*2)
            
            if round(trace[-1],6) not in unique_traces and not np.any(abs(np.diff(trace))>5):
                unique_traces = np.append(unique_traces, round(trace[-1],6))
                print('plotting...')    
                ax.fill_between(x_range[:, predictor_id], 
                                      lower, 
                                      higher, 
                                      alpha = 0.25, 
                                      facecolor = c
                                      )
                ax.plot(x_range[:, predictor_id], 
                        trace, 
                        color = c,
                        linewidth = 1.3,
                        linestyle = lnst,
                        alpha = 1,
                        # label = lbl
                        )
                print(f"{lbl}  {trace[0]/np.pi:.2f}±{(higher[0]-lower[0])/(2*np.pi):.2f}π")
                print(f"{lbl}  {trace[-1]/np.pi:.2f}±{(higher[-1]-lower[-1])/(2*np.pi):.2f}π")
                print(f"{lbl}  average effect size: {abs(trace[-1]-trace[0])/np.pi:.2f}")
            
            # for stats
            if trace[-1] > ylim[0] and trace[-1] < ylim[1] and trace[-1] not in last_vals:
                last_vals.append(trace[-1])
    
    # -------------------------------STATS-----------------------------------
    stat_dict = treadmill_circGLM.get_circGLM_stats(
            predictors = predictorlist,
            yyyymmdd = yyyymmdd,
            limb = limb,
            ref = ref,
            samples = samples,
            interaction = interaction,
            appdx = appdx,
            datafrac = datafrac,
            categ_var=categ_var,
            slopes = slopes,
            outputDir = Config.paths['mtTreadmill_output_folder'],
            iterations = iters,
            sBA_split_str = sba_str,
            mice = Config.mtTreadmill_config['mice_level']
                    ) 

    cont_coef_str = f"pred{predictor_id+1}"
    
    ax.text(xlim[0] + (0.05 * (xlim[1]-xlim[0])),
            ylim[1] - (0.05* (ylim[1]-ylim[0])),
            f"{predictorlist_str[predictor_id]}: {stat_dict[cont_coef_str]}",
            # color=c,
            fontsize=5)
    
    cat_stat = stat_dict[f'pred{len(predictorlist)+1}Llead']=='*' and stat_dict[f'pred{len(predictorlist)+1}Rlead']=='*'
    cat_stat_str = '*' if cat_stat else 'n.s.'
    ax.text(xlim[0] + (0.05 * (xlim[1]-xlim[0])),
            ylim[1] - (0.15* (ylim[1]-ylim[0])),
            f"RH phase: {cat_stat_str}",
            fontsize=5)
    # -------------------------------STATS-----------------------------------
    
    ax.set_title(tlt)
        
    # axes 
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xticks(xticks[::2])
    ax.set_xlabel(f"{xlabel}†")
    
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel('Left homolateral phase\n(rad)')
    
    # -------------------------------LEGEND----------------------------------- 
    # fig.legend(loc = 'center right', bbox_to_anchor=(1,0.65), fontsize=5)
    # -------------------------------LEGEND----------------------------------- 
       
    plt.tight_layout()
    
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    figtitle = f"mtTreadmill_{limb}_ref{ref}_{'_'.join(predictorlist)}_SLOPE{''.join(slopes)}_{interaction}_{appdx}_AVERAGE.svg"
    savepath = os.path.join(FigConfig.paths['savefig_folder'], figtitle)
    plt.savefig(savepath, 
                dpi = 300, 
                bbox_inches = 'tight',
                transparent = True)
    print(f"FIGURE SAVED AT {savepath}")
if __name__=="__main__":
    plot_figS8B()        
    
        
