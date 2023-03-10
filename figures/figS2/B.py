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
df, yyyymmdd = data_loader.load_processed_data(dataToLoad = 'locomParamsAcrossMice', 
                                               yyyymmdd = yyyymmdd,
                                               appdx = '_COMBINED')

for m in np.setdiff1d(np.unique(df['mouseID']), Config.passiveOpto_config['mice']):
    df = df[df['mouseID'] != m]

df['stimFreq_num'] = [int(x[:-2]) for x in df['stimFreq']]

subplot_num = 2
fig, ax = plt.subplots(1,subplot_num, figsize = (3,1.6))
for trialType, clr, lnst in zip(['headHeight', 'incline'],
                                [FigConfig.colour_config['neutral'], FigConfig.colour_config['main']], 
                                ['dashed', 'solid']):
    for i, (colname, label) in enumerate(zip(['medianSpeed', 'maxSpeed'], 
                                             ['Median speed (cm/s)', 'Maximum speed (cm/s)'],
                                             )):
        df_ttsub = df[df['trialType']==trialType]
        mouse_colours = [clr]*np.unique(df_ttsub['mouseID']).shape[0]
        sns.lineplot(x = 'stimFreq_num', 
                     y = colname, 
                     hue = 'mouseID', 
                     data = df_ttsub,
                     linewidth = 0.5, 
                     ci=None, 
                     palette=mouse_colours, 
                     ax = ax[i], 
                     linestyle = lnst,
                     alpha = 0.3)
    
        ax[i].set_xlabel('Stimulation\nfrequency (Hz)')
        ax[i].set_ylabel(label)
        ax[i].set_xlim(5, 50)
        ax[i].set_xticks([10,20,30,40,50])
        try:
            ax[i].get_legend().remove()  
        except:
            pass

        mod_types = ['linear', 'quadratic']
        mod_AIC = pd.read_csv(os.path.join(Config.paths['passiveOpto_output_folder'], f"{yyyymmdd}_mixedEffectsModel_AICRsq_{colname}_v_stimFreq_COMBINED.csv"))
        AICs = [mod_AIC['Value'].loc[np.where((mod_AIC['Model']==mod_types[0].capitalize()) & (mod_AIC['Metric']=='AIC'))[0][0]],
                mod_AIC['Value'].loc[np.where((mod_AIC['Model']==mod_types[0].capitalize()) & (mod_AIC['Metric']=='AIC'))[0][0]]]
        
        selected_type = mod_types[np.argmin(AICs)]
        
        mod = pd.read_csv(os.path.join(Config.paths['passiveOpto_output_folder'], f"{yyyymmdd}_mixedEffectsModel_{selected_type}_{colname}_v_stimFreq_COMBINED.csv"))
        
        if trialType == 'headHeight':
            print(f"{label}: {mod['Estimate'][1]} ± {mod['Std. Error'][1]}")
            print(f"{label} incline effect : {mod['Estimate'][2]} ± {mod['Std. Error'][2]}")
            
        x_centered = np.asarray( df['stimFreq_num'])-np.nanmean(df['stimFreq_num'])

        x_pred = np.linspace(x_centered.min(), x_centered.max(), endpoint=True)
        if trialType == 'headHeight':
            if selected_type == 'linear':
                y_pred = mod['Estimate'][0] + mod['Estimate'][1] * x_pred + np.nanmean(df[colname])
            elif selected_type == 'quadratic':
                y_pred = mod['Estimate'][0] + mod['Estimate'][1]*x_pred + mod['Estimate'][2]*x_pred**2 + np.nanmean(df[colname])    
        else:
            if selected_type == 'linear':
                y_pred = mod['Estimate'][0] + mod['Estimate'][1] * x_pred + mod['Estimate'][2] + np.nanmean(df[colname])
            elif selected_type == 'quadratic':
                y_pred = mod['Estimate'][0] + mod['Estimate'][1]*x_pred + mod['Estimate'][2]*x_pred**2 + mod['Estimate'][3] + np.nanmean(df[colname])
       
        x_pred += np.nanmean(df['stimFreq_num'])
        ax[i].plot(x_pred, 
                   y_pred, 
                   linewidth=1, 
                   color = clr, 
                   linestyle = lnst)
        
        # p-values
        for x, title in enumerate(['frequency', 'trial type']):
            p_text = title + ' ' + ('*' * (mod['Pr(>|t|)'][x+1] < FigConfig.p_thresholds).sum())
            if (mod['Pr(>|t|)'][x+1] < FigConfig.p_thresholds).sum() == 0:
                p_text += "n.s."
            ax[i].text(0.47,
                    1.15-(0.13*x), 
                    p_text, 
                    ha = 'center', 
                    color = FigConfig.colour_config['neutral'], 
                    transform=ax[i].transAxes)
        
ax[0].set_ylim(0,120)
ax[0].set_yticks(np.linspace(0,120,5,endpoint=True))
ax[1].set_ylim(60,220)
ax[1].set_yticks(np.linspace(60,220,5,endpoint=True))
# ax[2].set_ylim(-0.05,1)
#ax[3].set_ylim(0,1)

plt.tight_layout(w_pad = 2.8, pad = 0, h_pad = 0)
# plt.tight_layout()

lgd = fig.legend([([FigConfig.colour_config['homolateral'][2]],"solid"), 
                ([FigConfig.colour_config['greys'][2]],"dashed")], 
                ["Incline trials", "Head height trials"], 
                handler_map={tuple: AnyObjectHandler()}, 
                loc = 'upper center', bbox_to_anchor=(0.15,-0.25,0.7,0.2),
                mode="expand", borderaxespad=0, ncol = 2)

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"{yyyymmdd}_locomParamsAcrossMice_duration_speed_latency_vs_stimFreq_COMBINED.svg",
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight')


 
