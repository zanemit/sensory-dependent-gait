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
        "forelimbs": "#df9b9b", # salmon
        "hindlimbs": "#a3d4d7", # light teal
        "headbars": "#7a7a7a", # dark grey
        "main": "#377a7b", # teal
        "neutral": "#bdbdbd", # light grey
        "homologous" : ['#551B1B', '#882A2A', '#BC3939', '#D06767', '#DF9B9B'], # red tones
        "homolateral": ['#234B4D', '#377A7B', '#4CA7A9', '#75BFC2', '#A3D4D7'], # turquoise tones
        "diagonal": ['#4A4D23', '#777C37', '#A5AB4A', '#BDC374', '#D3D7A3'], # lime tones
        "greys": ['#383838', '#595959', '#7A7A7A', '#9C9C9C', '#BDBDBD'],
        "greys7": ['#CFCFCF', '#B2B2B2', '#969696', '#7A7A7A', '#5E5E5E', '#424242', '#262626'],
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
        if len(orig_handle[0])>2:
            l2 = plt.Line2D([x0,y0+width], [0.2*height,0.2*height], linewidth = 1,
                               linestyle = orig_handle[1], color=orig_handle[0][1])
            l3 = plt.Line2D([x0,y0+width], [0.8*height,0.8*height], linewidth = 1,
                               linestyle = orig_handle[1], color=orig_handle[0][2])
            return [l1, l2, l3]
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


