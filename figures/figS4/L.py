import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import os
from processing import utils_math, treadmill_circGLM
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def plot_figS4L():
    predictorlist = ['speed', 'snoutBodyAngle', 'incline','weight']#['speed', 'snoutBodyAngle', 'weight']
    predictorlist_str = ['speed', 'snoutBodyAngle', 'slope', 'weight']#['speed', 'snout-hump angle', 'weight']
    predictor = 'weight'
    predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
    appdx =  '_incline'#''
    tlt = 'Slope trials'
    yyyymmdd = '2022-08-18'
    slopes = ['pred2', 'pred3']
    limb = 'homolateral0'
    ref = 'COMBINEDcb'
    interaction = 'TRUEsecondary'#'TRUE'
    samples =  14949#13206
    datafrac = 1
    iters = 1000
    sba_split_str = 'sFALSE'
    
    
    ### PLOTTING
    ylim = (0.3*np.pi,1.5*np.pi)
    yticks = [0.5*np.pi,np.pi,1.5*np.pi]
    yticklabels = ["0.5π", "π", "1.5π"]  
    xlim, xticks, xlabel = treadmill_circGLM.get_predictor_range(predictor)
    
    fig, ax = plt.subplots(1,1,figsize = (1.35,1.35)) #1.6,1.4 for 4figs S2 bottom row
    
    last_vals = [] # for stats
        
    x_range, phase_preds = treadmill_circGLM.get_circGLM_slopes(
            predictors = predictorlist,
            yyyymmdd = yyyymmdd,
            limb = limb,
            ref = ref,
            categ_var='refLimb',
            samples = samples,
            interaction = interaction,
            appdx = appdx,
            datafrac = datafrac,
            slopes = slopes,
            outputDir = Config.paths['passiveOpto_output_folder'],
            iterations = iters,
            mice = Config.passiveOpto_config['mice'],
            sBA_split_str = sba_split_str
                    ) 
    
    c = FigConfig.colour_config['homolateral'][1]
    pp = phase_preds[:, :, predictor_id, 0, 0]
    
    # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
    unique_traces = np.empty((0))
    for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
        print(f"Working on data range {k}...")
        if k == 1:
            pp[pp<0] = pp[pp<0]+2*np.pi
        if k == 2:
            pp[pp<np.pi] = pp[pp<np.pi]+2*np.pi
            
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
            ax.fill_between(x_range[:,predictor_id], 
                                  lower, 
                                  higher, 
                                  alpha = 0.25, 
                                  facecolor = c
                                  )
            ax.plot(x_range[:,predictor_id], 
                    trace, 
                    color = c,
                    linewidth = 1,
                    alpha = 1,
                    # label = lbl
                    )
        
        # for stats
        if trace[-1] > ylim[0] and trace[-1] < ylim[1] and trace[-1] not in last_vals:
            last_vals.append(trace[-1])
    
    # -------------------------------STATS-----------------------------------
    stat_dict = treadmill_circGLM.get_circGLM_stats(
            predictors = predictorlist,
            yyyymmdd = yyyymmdd,
            limb = limb,
            ref = ref,
            categ_var='refLimb',
            samples = samples,
            interaction = interaction,
            appdx = appdx,
            datafrac = datafrac,
            slopes = slopes,
            outputDir = Config.paths['passiveOpto_output_folder'],
            iterations = iters,
            mice = Config.passiveOpto_config['mice'],
            sBA_split_str = sba_split_str
                    ) 
    
    cont_coef_str = f"pred{predictor_id+1}"
    
    ax.text(xticks[1] + (0.35 * (xlim[1]-xticks[1])),
            ylim[1] - (0.05* (ylim[1]-ylim[0])),
            f"{predictorlist_str[predictor_id]}: {stat_dict[cont_coef_str]}",
            # color=c,
            fontsize=5)
    
    # -------------------------------STATS-----------------------------------
    
    ax.set_title(tlt)
        
    # axes 
    ax.set_xlim(xticks[1], xlim[1])
    ax.set_xticks(xticks[1::2])
    ax.set_xlabel("Body weight\n(g)")
    
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel('Homolateral phase\n(rad)')
    
    # -------------------------------LEGEND----------------------------------- 
    # fig.legend(loc = 'center right', bbox_to_anchor=(1,0.65), fontsize=5)
    # -------------------------------LEGEND----------------------------------- 
       
    plt.tight_layout()
    
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    figtitle = f"passiveOpto_weight_inclineTrials.svg"
    savepath = os.path.join(FigConfig.paths['savefig_folder'], figtitle)
    plt.savefig(savepath, 
                dpi = 300, 
                bbox_inches = 'tight',
                transparent = True)
    print(f"FIGURE SAVED AT {savepath}")

if __name__=="__main__":
    plot_figS4L()
        
    
        
