import sys
from pathlib import Path
import pandas as pd
import os
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_processing, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

phase_bounds = [0.1, 0.4, 0.6, 0.9]
clr = '#62b7b7'
clr_bg = '#d6ecec'
fontsize = 6

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(0.8,0.8))
ax.set_ylim(0,1)

for i, (phb, lbl) in enumerate(zip(phase_bounds,
                    ['right-\nleading', 'alternation', 'left-\nleading', 'synchrony']
                    )):
    ang = phb*2*np.pi
    ax.plot([ang,ang], [0,1], color=clr, lw=1.6)
    ax.text(ang, 1.4, f"{phb*2:.1f}π",
            ha='center', va='center', color=clr)
    ax.text((1+i)*(0.5*np.pi), 1.05, lbl, color='black',
            ha='center', va='center', fontsize=fontsize)
    
ax.set_xticks([])
ax.set_yticks([])
ax.spines['polar'].set_linewidth(1.6)
ax.spines['polar'].set_color(clr)
ax.set_facecolor(clr_bg)

fig.savefig(Path(FigConfig.paths['savefig_folder']) / f"phase_bounds_polar.svg",
            transparent = False,
            dpi =300)