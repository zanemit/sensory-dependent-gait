import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensoryDependentGait")

from preprocessing import data_loader
from preprocessing.data_config import Config
from figures.fig_config import Config as FigConfig
from figures import scalebars
from figures.fig_config import AnyObjectHandler


bp_align = 'postF2'
level_dict1 = {'deg-40': 40, 'deg-20': 20, 'deg0': 0, 'deg20': -20, 'deg40': -40}
level_dict2 = {'rl17': -15, 'rl12': -10, 'rl7': -5, 'rl2': 0, 'rl-3': 5, 'rl-8': 10, 'rl-13': 15}

# plot average (across mice) posture polygons (one per level)
fig, axes = plt.subplots(3,1, figsize = (1.4,2.3), sharex = True)

for yyyymmdd, ax, title, param, level_dict in zip(['2021-10-26', '2022-04-02', '2022-04-04'], 
                                                  [axes[0], axes[1], axes[2]],
                                                  ['Head height trials','Incline trials (low)','Incline trials (medium)'], 
                                                  ['headHW', 'levels','levels'],
                                                  [level_dict2, level_dict1, level_dict1]):
    
    meanParamDF, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                        dataToLoad='meanParamDF', 
                                                        yyyymmdd = yyyymmdd,
                                                        appdx = f'_{param}')
        
    dlcPostureX, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                     dataToLoad = 'dlcPostureX', 
                                                     yyyymmdd = yyyymmdd)
    
    dlcPostureY, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                     dataToLoad = 'dlcPostureY', 
                                                     yyyymmdd = yyyymmdd)
    dlcPosts, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"], 
                                                  dataToLoad = 'dlcPosts', 
                                                  yyyymmdd = yyyymmdd)
    
    bps_posture_polygon = ['b2','b3','body','b4','b5','b6','b7','b8','tailbase','rH2','rH1','rF2','rF1','snout','b2b']
    bps_posts = ['postF1', 'postF2', 'postH1', 'postH2']
    mice = np.unique(dlcPostureX.columns.get_level_values(0))
    levels = np.unique(dlcPostureX.columns.get_level_values(1))
    trials = np.unique(dlcPostureX.columns.get_level_values(2))
    
    level_dict_inv = {v: k for k,v in level_dict.items()}
    
    if param == 'levels':
        levels = np.sort(list(level_dict_inv.keys()))
    elif param == 'headHW':
        fpheadHW, _ = data_loader.load_processed_data(outputDir = Config.paths["forceplate_output_folder"],
                                                      dataToLoad = 'forceplateHeadHW', 
                                                      yyyymmdd = yyyymmdd)
        param_num = 4
        param_split = np.linspace(np.asarray(fpheadHW).min()-0.0001, np.asarray(fpheadHW).max(), param_num+1)
        fpheadHW_df = fpheadHW.unstack().reset_index()
        df_grouped = meanParamDF.groupby(pd.cut(fpheadHW_df[0], param_split)) 
        group_row_ids = df_grouped.groups
        mouse_dict = {m : im for im,m in enumerate(mice)}
        fpheadHW_df['param'] = [level_dict[x] for x in fpheadHW_df['level']]
        levels = group_row_ids.keys()

        # prepare data to be run in the same loop as 'snoutBodyAngle'
        working_df = fpheadHW_df
    
    mouse_coords = np.empty((len(mice), len(bps_posture_polygon), len(levels), len(trials), 2)) # the final dimension is for x and y coords           
    mouse_coords[:] = np.nan
  
    pd.options.mode.chained_assignment = None
    
    if param == 'levels':
        for im, m in enumerate(mice):
            for ilvl, lvl in enumerate(levels):
                for itr, tr in enumerate(trials):
                    try:
                        # process body and post coordinates
                        dlcX_sub = dlcPostureX.loc[:, (m, level_dict_inv[lvl], tr)]
                        dlcX_sub.loc[:,'b2b'] = dlcX_sub['b2']
                    
                        dlcY_sub = dlcPostureY.loc[:, (m, level_dict_inv[lvl], tr)]
                        dlcY_sub.loc[:,'b2b'] = dlcY_sub['b2']
        
                        dlcPosts_subX = dlcPosts.loc[:, (m, level_dict_inv[lvl], tr, slice(None), 'x')]
                        dlcPosts_subY = dlcPosts.loc[:, (m, level_dict_inv[lvl], tr, slice(None), 'y')]
                        
                         # align data if necessary
                        mouse_coords[im, :, ilvl, itr, 0] = np.nanmean(dlcX_sub, axis = 0)- np.nanmean(dlcPosts_subX.loc[:, (m, level_dict_inv[lvl], tr, bp_align, 'x')])
                        mouse_coords[im, :, ilvl, itr, 1] = np.nanmean(dlcY_sub, axis = 0)- np.nanmean(dlcPosts_subY.loc[:, (m, level_dict_inv[lvl], tr, bp_align, 'y')])
                         
                        meanParamDF_sub = meanParamDF[(meanParamDF['mouse'] == m) & (meanParamDF['param'] == lvl)]
                    except:
                        continue
                    
    elif param == 'headHW':
        for ik, key in enumerate(levels):
            for row in group_row_ids[key]:
                m = working_df.iloc[row,:]['mouse']
                im = mouse_dict[m]
                lvl = working_df.iloc[row,:]['param']
                for itr, tr in enumerate(trials):
                    # try:
                        # process body and post coordinates
                        if (m, level_dict_inv[lvl], tr) in dlcPostureX.columns:
                            dlcX_sub = dlcPostureX.loc[:, (m, level_dict_inv[lvl], tr)]
                            dlcX_sub.loc[:,'b2b'] = dlcX_sub['b2']
                        
                            dlcY_sub = dlcPostureY.loc[:, (m, level_dict_inv[lvl], tr)]
                            dlcY_sub.loc[:,'b2b'] = dlcY_sub['b2']

                            dlcPosts_subX = dlcPosts.loc[:, (m, level_dict_inv[lvl], tr, slice(None), 'x')]
                            dlcPosts_subY = dlcPosts.loc[:, (m, level_dict_inv[lvl], tr, slice(None), 'y')]
                            
                             # align data if necessary
                            mouse_coords[im, :, ik, itr, 0] = np.nanmean(dlcX_sub, axis = 0)- np.nanmean(dlcPosts_subX.loc[:, (m, level_dict_inv[lvl], tr, bp_align, 'x')])
                            mouse_coords[im, :, ik, itr, 1] = np.nanmean(dlcY_sub, axis = 0)- np.nanmean(dlcPosts_subY.loc[:, (m, level_dict_inv[lvl], tr, bp_align, 'y')])
                            
                            meanParamDF_sub = meanParamDF[(meanParamDF['mouse'] == m) & (meanParamDF['param'] == lvl)]

                        else:
                            print(f"{im}, {m}, {ik}, {key}, {itr}, {tr} does not have data?")
    
    # draw rectangles for posts
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0,0), 96, -52, color = FigConfig.colour_config['neutral']))
    ax.add_patch(Rectangle((-260,0), 96, -52, color = FigConfig.colour_config['neutral']))
    
    # plot sillhouettes
    for ilvl, level in enumerate(levels):
        # average across trials (axis=2), then across mice (axis = 0); if mouse is set, trial axis is axis =1 
        ax.fill(np.nanmean(mouse_coords[:, :, ilvl,:,0], axis =2 ).T, 
                np.nanmean(mouse_coords[:, :, ilvl,:,1], axis = 2).T,  
                color = FigConfig.colour_config['homolateral'][ilvl], 
                alpha = 0.1)
        
        if param == 'levels':
            lbl = f"{level} deg"
        elif param == 'headHW':
            lbl = f"({level.left:.2f}, {level.right:.2f}]"
            
        ax.plot(np.nanmean(np.nanmean(mouse_coords[:, :, ilvl,:,0], axis = 2), axis = 0), 
                np.nanmean(np.nanmean(mouse_coords[:, :, ilvl,:,1], axis = 2), axis = 0), 
                color =  FigConfig.colour_config['homolateral'][ilvl], 
                linestyle = '-', 
                linewidth = 1, 
                zorder = 6,
                label = lbl
                )

    ax.set_ylim(-200,200)
    ax.set_xlim(-400,240)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-230,0,230])
    ax.set_yticks([-230,0,230])
    ax.set_title(title)

    scalebars.add_scalebar(ax, matchx=True, matchy=False, hidex=True, hidey=True, 
                            loc = 'lower left', pad = 1, labelx = "3 cm")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels,handles))

    if param == 'levels':
        bbox_coords = (0.88,0.3,0.4,0.1)
        lgd_tlt = 'Incline'
    elif param == 'headHW':
        bbox_coords = (0.88,0.75,0.4,0.1)
        lgd_tlt = 'Head height'
    
    if ax == axes[0]:
        lgd = fig.legend(handles = by_label.values(), labels = by_label.keys(), 
                         handlelength = 0, title = lgd_tlt,
                        loc = 'center left', bbox_to_anchor=bbox_coords,
                        mode="expand", borderaxespad=0, ncol = 1,
                        borderpad = 0)

        for line, text in zip(lgd.get_lines(), lgd.get_texts()):
            text.set_color(line.get_color())
            text.set_ha('left')
    
    elif ax == axes[2]:
        lgd2 = fig.legend(handles = by_label.values(), labels = by_label.keys(), 
                         handlelength = 0, title = lgd_tlt,
                        loc = 'center left', bbox_to_anchor=bbox_coords,
                        mode="expand", borderaxespad=0, ncol = 1,
                        borderpad = 0)
        for line, text in zip(lgd2.get_lines(), lgd2.get_texts()):
            text.set_color(line.get_color())
            text.set_ha('right')
        ax.add_artist(lgd2)

    fig.tight_layout(h_pad=-0.1)

fig.savefig(os.path.join(FigConfig.paths['savefig_folder'], yyyymmdd + '_sillhouette_headLVL_incline.svg'), 
            bbox_extra_artists = (lgd, ), 
            bbox_inches = 'tight',
            transparent = True)
