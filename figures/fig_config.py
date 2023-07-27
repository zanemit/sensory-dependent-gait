from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerBase

class Config:
    rcParams['figure.dpi']= 300

    rcParams['axes.labelsize']=6
    rcParams['axes.labelpad']=0.5
    rcParams['axes.titlesize']=6
    rcParams['axes.xmargin']=0
    rcParams['axes.ymargin']=0
    rcParams['axes.facecolor']=[1,1,1,0] 
    rcParams['axes.linewidth']=0.5 
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.titlesize'] = 6
    
    rcParams['lines.linewidth']=0.5

    rcParams['figure.subplot.wspace'] = 0.5
    rcParams['figure.titlesize'] = 6
    
    rcParams['xtick.major.width']=0.5 #tick width
    rcParams['ytick.major.width']=0.5
    rcParams['xtick.major.size']=4 #tick length
    rcParams['ytick.major.size']=4
    rcParams['xtick.labelsize']=6
    rcParams['ytick.labelsize']=6
    
    rcParams['font.size']=6
    
    rcParams['legend.fontsize']=6
    rcParams['legend.facecolor'] = [1,1,1,0]
    rcParams['legend.edgecolor'] = '#cecece'
    rcParams['legend.frameon'] = True
    
    rcParams['patch.linewidth'] = 0.5
    
    paths = {
        "savefig_folder": r"C:\Users\MurrayLab\Documents\PaperFIGURES\figures"
        }
    
    p_thresholds = [0.05, 0.01, 0.001]
    
    colour_config = {
        "forelimbs": "#cc6677", # salmon
        "hindlimbs": "#44aa99", # light teal
        "headbars": "#878787", # dark grey
        "main": "#44aa99", # teal
        "neutral": '#969696', # light grey
        "homologous" : ['#993344', '#bf4055', '#cc6677', '#d98c99', '#e5b3bb'], # red tones
        "homolateral": ['#276358', '#368779', '#44aa99', '#62c0b0', '#87cfc2'], # turquoise tones
        "diagonal": ['#c0a830', '#d3bd50', '#ddcc77', '#e8dca1', '#f2ebca'], # lime tones
        "greys": ['#878787', '#a1a1a1', '#bbbbbb', '#d4d4d4', '#ededed'],
        "greys7": ['#262626','#424242','#7A7A7A', '#969696','#B2B2B2','#CFCFCF'],#'#5E5E5E',
        "mains8": ['#102323', '#21494A', '#326E6F', '#439495', '#59B2B4', '#7FC4C5', '#A4D5D6', '#CBE7E7']
        }

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], linewidth = 1,
                           linestyle=orig_handle[1], color=orig_handle[0][0])
        if len(orig_handle[0])==2:
            l2 = plt.Line2D([x0,y0+width], [0.2*height,0.2*height], linewidth = 1,
                               linestyle = orig_handle[1], color=orig_handle[0][1])
            return [l1, l2]
        if len(orig_handle[0])==3:
            l2 = plt.Line2D([x0,y0+width], [0.2*height,0.2*height], linewidth = 1,
                               linestyle = orig_handle[1], color=orig_handle[0][1])
            l3 = plt.Line2D([x0,y0+width], [0.8*height,0.8*height], linewidth = 1,
                               linestyle = orig_handle[1], color=orig_handle[0][2])
            return [l1, l2, l3]
        if len(orig_handle[0])>3:
            l2 = plt.Line2D([x0,y0+width], [0.2*height,0.2*height], linewidth = 1,
                               linestyle = orig_handle[1], color=orig_handle[0][1])
            l3 = plt.Line2D([x0,y0+width], [0.8*height,0.8*height], linewidth = 1,
                               linestyle = orig_handle[1], color=orig_handle[0][2])
            l4 = plt.Line2D([x0,y0+width], [1.1*height,1.1*height], linewidth = 1,
                               linestyle = orig_handle[1], color=orig_handle[0][3])
            return [l1, l2, l3, l4]
        else:
            return [l1]
        
class AnyObjectHandlerDouble(HandlerBase):        
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], linewidth = 1,
                           linestyle=orig_handle[1][0], color=orig_handle[0][0])
        if len(orig_handle[0])==2 and len(orig_handle[1])==2:
            l2 = plt.Line2D([x0,y0+width], [0.2*height,0.2*height], linewidth = 1,
                               linestyle = orig_handle[1][1], color=orig_handle[0][1])
            return [l1, l2]
        if len(orig_handle[0])==3 and len(orig_handle[1])==3:
            l2 = plt.Line2D([x0,y0+width], [0.2*height,0.2*height], linewidth = 1,
                               linestyle = orig_handle[1][1], color=orig_handle[0][1])
            l3 = plt.Line2D([x0,y0+width], [0.8*height,0.8*height], linewidth = 1,
                               linestyle = orig_handle[1][2], color=orig_handle[0][2])
            return [l1, l2, l3]
        if len(orig_handle[0])==4 and len(orig_handle[1])==4:
            l2 = plt.Line2D([x0,y0+width], [0.2*height,0.2*height], linewidth = 1,
                               linestyle = orig_handle[1][1], color=orig_handle[0][1])
            l3 = plt.Line2D([x0,y0+width], [0.8*height,0.8*height], linewidth = 1,
                               linestyle = orig_handle[1][2], color=orig_handle[0][2])
            l4 = plt.Line2D([x0,y0+width], [-0.1*height,-0.1*height], linewidth = 1,
                               linestyle = orig_handle[1][3], color=orig_handle[0][3])
            return [l1, l2, l3, l4]
        else:
            return [l1]


