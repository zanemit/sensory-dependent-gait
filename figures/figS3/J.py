import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures.fig_config import AnyObjectHandler

yyyymmdd = '2022-08-18'
colname = 'medianSpeed'
param = 'snoutBodyAngle'
group_num = 5

df, yyyymmdd = data_loader.load_processed_data(dataToLoad = 'locomParamsAcrossMice', 
                                               yyyymmdd = yyyymmdd,
                                               appdx = '_COMBINED')

# subset mice
for m in np.setdiff1d(np.unique(df['mouseID']), Config.passiveOpto_config['mice']):
    df = df[df['mouseID'] != m]
    

df['headLVL'] = [-int(h[3:]) if 'deg' in h else h for h in df['headLVL']]
df['stimFreq_num'] = [int(x[:-2]) for x in df['stimFreq']]

fig, ax = plt.subplots(1,1, figsize = (1.4,1.5))
for ix, (trialType, clr, lnst, p_tlt, appdx) in enumerate(zip(
                                             ['incline','headHeight'],
                                             [FigConfig.colour_config['main'],FigConfig.colour_config['neutral']], 
                                             ['solid', 'dashed'],
                                             ['Incline trials', 'Head height trials'],
                                             ['_incline', ''])):
    df_tt = df[df['trialType']==trialType]
    xvals_all = []
    for im, m in enumerate(np.unique(df['mouseID'])):
        df_tt_m = df_tt[df_tt['mouseID'] == m]
        param_split = np.linspace(df_tt_m[param].min()-0.0001, df_tt_m[param].max(), group_num+1)
        xvals = [np.mean((a,b,)) for a,b in zip(param_split[:-1], param_split[1:])]
        df_grouped = df_tt_m.groupby(pd.cut(df_tt_m[param], param_split)) 
        group_row_ids = df_grouped.groups
        
        yvals = [np.nanmean(df_tt_m.loc[val,colname].values) for key,val in group_row_ids.items()]
       
        ax.plot(xvals, 
                      yvals,   
                      alpha=0.3, 
                      linewidth = 0.5, 
                      color = clr, 
                      linestyle = lnst)
        
        xvals_all.append(np.min(xvals))
        xvals_all.append(np.max(xvals))   
    
    ax.set_ylim(0,120)
    ax.set_yticks(np.linspace(0,120,7,endpoint=True))
            
    mod_types = ['linear', 'quadratic']
    mod_AIC = pd.read_csv(os.path.join(Config.paths['passiveOpto_output_folder'], f"{yyyymmdd}_mixedEffectsModel_AICRsq_{colname}_v_stimFreq_{param}{appdx}.csv"))
    AICs = [mod_AIC['Value'].loc[np.where((mod_AIC['Model']==mod_types[0].capitalize()) & (mod_AIC['Metric']=='AIC'))[0][0]],
            mod_AIC['Value'].loc[np.where((mod_AIC['Model']==mod_types[1].capitalize()) & (mod_AIC['Metric']=='AIC'))[0][0]]]
    
    selected_type = mod_types[np.argmin(AICs)]
    
    mod = pd.read_csv(os.path.join(Config.paths['passiveOpto_output_folder'], f"{yyyymmdd}_mixedEffectsModel_{selected_type}_{colname}_v_stimFreq_{param}{appdx}.csv"))

    x_centered = np.asarray(df_tt[param])-np.mean(df_tt[param])
    mean_stimFreq = np.nanmean(df_tt['stimFreq_num'])

    x_pred = np.linspace(np.min(xvals_all)-np.mean(df_tt[param]), np.max(xvals_all)-np.mean(df_tt[param]), endpoint=True)
    # if param == 'headLVL' or trialType == 'headHeight':
    if selected_type == 'linear':
        y_pred = mod['Estimate'][0] + mod['Estimate'][2] * x_pred + np.nanmean(df[colname])
    elif selected_type == 'quadratic':
        y_pred = mod['Estimate'][0] + mod['Estimate'][2]*x_pred + mod['Estimate'][3]*x_pred**2 + np.nanmean(df[colname])    

    x_pred += np.mean(df_tt[param])
    ax.plot(x_pred, 
               y_pred, 
               linewidth=1, 
               color = clr, 
               linestyle = lnst)
    
    # p-values
    p_text = p_tlt + ' ' + ('*' * (mod['Pr(>|t|)'][2] < FigConfig.p_thresholds).sum())
    if (mod['Pr(>|t|)'][2] < FigConfig.p_thresholds).sum() == 0:
        p_text += "n.s."
    ax.text(0.52, 1.05-(0.13*ix), p_text, ha = 'center', 
            color = clr, 
            transform=ax.transAxes)

    
    # try:
    #     ax.get_legend().remove()  
    # except:
    #     pass

    
ax.set_xlabel('Snout-hump angle (deg)')
ax.set_xticks(np.linspace(140,180,5,endpoint = True))

ax.set_ylabel('Median speed (cm/s)')

plt.tight_layout(w_pad = 1, pad = 0, h_pad = 0)
# # plt.tight_layout()


fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_locomParamsAcrossMice_medSpeed_vs_stimFreq_snoutBodyAngle.svg", dpi=300)



 
