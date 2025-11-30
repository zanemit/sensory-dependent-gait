import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import os
from processing import treadmill_circGLM
from processing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import get_palette_from_html

def plot_figS5E():
    main_clr = FigConfig.colour_config['homologous'][1]
    palette = get_palette_from_html(main_clr,
                                    lightness_values=[0.6,0.65,0.7,0.75,0.8])
    
    #---------------HEAD HEIGHT TRIALS--------------------
    predictorlist = ['speed', 'snoutBodyAngle', 'incline']
    predictorlist_str = ['speed', 'snout-hump angle', 'incline']
    predictor = 'incline'
    predictor_id = np.where(np.asarray(predictorlist) == predictor)[0][0]
    appdx = '_incline'
    samplenum = 7726
    tlt = 'Slope trials'
    yyyymmdd = '2022-08-18' #'2024-09-11' #'2022-08-18'
    slopes = ['pred2', 'pred3']
    limb = 'rH0'
    datafrac = 1
    ref = 'lH1altadvancedblncd'
    interaction = 'TRUEthreeway'
    rfl_str = False
    sba_str = 'sBAsplitFALSE_FLIPPED'
    
    mice_unilateral_inj = Config.injection_config['right_inj_imp'] + Config.injection_config['left_inj_imp'] 
    mouselist = np.intersect1d(Config.passiveOpto_config['mice'], mice_unilateral_inj)

    #---------------HEAD HEIGHT TRIALS--------------------
    
    unique_traces = np.empty((0))
    
    ### PLOTTING
    ylim = (-0.5*np.pi,1.5*np.pi)
    yticks = [-0.5*np.pi, 0, 0.5*np.pi,np.pi, 1.5*np.pi]
    yticklabels = ["-0.5π", "0", "0.5π", "π", "1.5π"]  
    xlim, xticks, xlabel = treadmill_circGLM.get_predictor_range(predictor)
    
    #---------------PER MOUSE--------------------
    fig, ax = plt.subplots(1,1,figsize = (1.35,1.35)) #1.6,1.4 for 4figs S2 bottom row
    for i in range(1, len(mouselist)+1):
        x_range, phase_preds = treadmill_circGLM.get_circGLM_slopes_per_mouse(
                predictors = predictorlist,
                yyyymmdd = yyyymmdd,
                limb = limb,
                ref = ref,
                samples =samplenum,
                interaction = interaction,
                appdx = appdx,
                datafrac = datafrac,
                categ_var = None,
                slopes = slopes,
                outputDir = Config.paths['passiveOpto_output_folder'],
                iterations = 1000,
                mice = mouselist,
                sBA_split_str = sba_str
                        ) 
        
        c = np.random.choice(palette, 1)[0]
        pp = phase_preds[:, :, predictor_id, i, 0]
        xr = x_range[:, predictor_id, i, 0]
        print(i, xr[-1]-xr[0])
        # compute and plot mean phases for three circular ranges so that the plots look nice and do not have lines connecting 2pi to 0
        for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
            print(f"Working on data range {k}...")
            if k == 1:
                pp[pp<0] = pp[pp<0]+2*np.pi
            if k == 2:
                pp[pp<np.pi] = pp[pp<np.pi]+2*np.pi
                
            trace = scipy.stats.circmean(pp[:,:],high = hi, low = lo, axis = 0)
            
            if round(trace[-1],6) not in unique_traces and not np.any(abs(np.diff(trace))>5):
                unique_traces = np.append(unique_traces, round(trace[-1],6))
                print('plotting...')    
                
                ax.plot(xr, 
                        trace, 
                        color = c,
                        linewidth = 0.7, 
                        alpha = 0.8
                        )
    #---------------PER MOUSE--------------------
      
    #---------------AVERAGE--------------------            
    xr_avg, phase_preds_avg = treadmill_circGLM.get_circGLM_slopes(
            predictors = predictorlist,
            yyyymmdd = yyyymmdd,
            limb = limb,
            ref = ref,
            categ_var = None,
            samples = samplenum,
            interaction = interaction,
            appdx = appdx,
            datafrac = datafrac,
            slopes = slopes,
            outputDir = Config.paths['passiveOpto_output_folder'],
            iterations = 1000,
            mice = mouselist,
            sBA_split_str = sba_str
                    ) 
    
    unique_traces = np.empty((0))           
    pp_avg = phase_preds_avg[:, :, predictor_id, 0, 0]            
    for k, (lo, hi) in enumerate(zip([-np.pi, 0, np.pi] , [np.pi, 2*np.pi, 3*np.pi])):
        print(f"Working on data range {k}...")
        if k == 1:
            pp_avg[pp_avg<0] = pp_avg[pp_avg<0]+2*np.pi
        if k == 2:
            pp_avg[pp_avg<np.pi] = pp_avg[pp_avg<np.pi]+2*np.pi
            
        trace_avg = scipy.stats.circmean(pp_avg[:,:],high = hi, low = lo, axis = 0)
        
        if round(trace_avg[-1],6) not in unique_traces and not np.any(abs(np.diff(trace_avg))>5):
            print('plotting...')    
            
            ax.plot(xr_avg[:,predictor_id], 
                    trace_avg, 
                    color = main_clr,
                    linewidth = 1.5, 
                    alpha = 1
                    )
    #---------------AVERAGE--------------------     
    
    ax.set_title(tlt)
        
    # axes 
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xticks(xticks)
    ax.set_xlabel("Surface slope\n(deg)")
    
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel('Right hindlimb phase\n(rad)')
    
    # -------------------------------STATS-----------------------------------
    samples = 10923
    datafrac = 1 
    ref = 'lH1altadvanced'
    sba_str = 'sFLIPPED'
    categ_var = 'lF0_categorical'
    stat_dict = treadmill_circGLM.get_circGLM_stats(
            predictors = predictorlist,
            yyyymmdd = yyyymmdd,
            limb = limb,
            ref = ref,
            categ_var = categ_var,
            samples = samples,
            interaction = interaction,
            appdx = appdx,
            datafrac = datafrac,
            slopes = slopes,
            outputDir = Config.paths['passiveOpto_output_folder'],
            iterations = 1000,
            mice = Config.passiveOpto_config['mice'],
            sBA_split_str = sba_str
                    ) 
    
    cont_coef_str = f"pred{predictor_id+1}"
    
    ax.text(xlim[0] + (0.5 * (xlim[1]-xlim[0])),
            ylim[1] - (0.2* (ylim[1]-ylim[0])),
            f"{predictorlist_str[predictor_id]}:\n{stat_dict[cont_coef_str]}",
            ha='center',
            fontsize=5)
    # -------------------------------STATS-----------------------------------
        
    plt.tight_layout()
    
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    figtitle = f"passiveOpto_{yyyymmdd}_{appdx}_{limb}_ref{ref}_{'_'.join(predictorlist)}_SLOPE{''.join(slopes)}_{interaction}_{appdx}_PER_MOUSE.svg"
    savepath=os.path.join(FigConfig.paths['savefig_folder'], figtitle)
    plt.savefig(savepath, 
                dpi = 300,  
                bbox_inches = 'tight',
                transparent = True)
    print(f"FIGURE SAVED AT {savepath}")

if __name__=="__main__":       
    plot_figS5E()
        
