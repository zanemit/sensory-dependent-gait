import numpy as np
import pandas as pd
import os
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

def threeway_stats(yyyymmdd,
                   model_name,
                   conds,
                   model_type = 'quadratic',
                   appdx = '',
                   outputDir = Config.paths['passiveOpto_output_folder'],
                   ):
    """
    yyyymmdd (str): YYYY-MM-DD
    model_name (str): special name added to the model
    conds (list of strs): list of conditions
    model_type (str): 'quadratic'
    appdx (str): added identifier
    """
    
    statspath1 = os.path.join(outputDir, f"{yyyymmdd}_mixedEffectsModel_{model_type}_{model_name}_{appdx}.csv")
    statspath2 = os.path.join(outputDir, f"{yyyymmdd}_mixedEffectsModel_{model_type}_{model_name}_{appdx}_RELEVELED.csv")

    stats1 = pd.read_csv(statspath1, index_col = 0)
    stats2 = pd.read_csv(statspath2, index_col = 0)
    
    condnames = [f"setup{c}" for c in conds]

    # which conditions are in stats1?
    default = np.setdiff1d(condnames, np.asarray(stats1.index))[0].split("setup")[-1]
    others = [c.split("setup")[-1] for c in np.intersect1d(condnames, np.asarray(stats1.index))]

    p1 = stats1.loc[f"setup{others[0]}", "Pr(>|t|)"] # between default and others0
    p2 = stats1.loc[f"setup{others[1]}", "Pr(>|t|)"] # between default and others1

    # which param from others is also in stats2?
    condnames_rlvd = [f"setupReleveled{c}" for c in conds]
    default2 = np.setdiff1d(condnames_rlvd, np.asarray(stats2.index))[0].split("setupReleveled")[-1]
    nondefault = np.setdiff1d(others, np.asarray(default2))[0]
    p3 = stats2.loc[f"setupReleveled{nondefault}", "Pr(>|t|)"] # between the remaining variables

    p_arr = np.asarray([[p1, p2], [p3, np.nan]])
    # row0 name = default
    # col0 name = others0
    # col1 name = others1
    # row1 name = default2 if default2!=others[0] else nondefault
    row1name = default2 if default2 != others[0] else nondefault
    # p-values
    p_text = np.empty((2,2),dtype = object)
    i = 0; j = 0
    num_comp = 3
    for k in range(p_arr.shape[0] * p_arr.shape[1]):
        txt = '*' * (p_arr[i,j] < np.asarray(FigConfig.p_thresholds)/num_comp).sum()
        if not np.isnan(p_arr[i,j]) and (p_arr[i,j] < np.asarray(FigConfig.p_thresholds)/num_comp).sum() == 0:
            txt += "n"
        p_text[i, j] = txt
        i = (k+1)//p_arr.shape[1]
        j = (k+1)%p_arr.shape[1]

    pseudo_arr = np.asarray([[0,0],[0,np.nan]])
    labels = ([others[0], others[1]], [default, row1name])
    
    return p_arr, p_text, labels, pseudo_arr
    
    