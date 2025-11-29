from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
from figures.fig_config import Config as FigConfig

def plot_figS2D():
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
    
    # save fig
    os.makedirs(FigConfig.paths['savefig_folder'], exist_ok=True)
    savepath = Path(FigConfig.paths['savefig_folder']) / "phase_bounds_polar.svg"
    fig.savefig(savepath,
                transparent = False,
                dpi =300)
    print(f"FIGURE SAVED AT {savepath}")

if __name__=="__main__":
    plot_figS2D()