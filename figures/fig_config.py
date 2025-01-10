from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerBase
import string
import colorsys

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
        "savefig_folder": r"C:\Users\MurrayLab\Documents\PaperFIGURES\figures\updated_ms"
        }
    
    p_thresholds = [0.05, 0.01, 0.001]
    
    colour_config = {
        "forelimbs": "#cc6677", # salmon
        "hindlimbs": "#44aa99", # light teal
        "headbars": "#878787", # dark grey
        "main": "#44aa99", # teal
        "neutral": '#969696', # light grey
        "reference": ['#373092', '#453db8', '#665eca', '#8a84d7', '#afabe3'], # purple tones
        "homologous" : ['#993344', '#bf4055', '#cc6677', '#d98c99', '#e5b3bb'], # red tones
        "homolateral": ['#276358', '#368779', '#44aa99', '#62c0b0', '#87cfc2'], # turquoise tones
        "diagonal": ['#978326', '#c0a830', '#d3bd50', '#ddcc77', '#e8dca1'], # lime tones
        "greys": ['#878787', '#a1a1a1', '#bbbbbb', '#d4d4d4', '#ededed'],
        "greys7": ['#262626','#424242','#7A7A7A', '#969696','#B2B2B2','#CFCFCF','#5E5E5E'],
        "mains8": ['#102323', '#21494A', '#326E6F', '#439495', '#59B2B4', '#7FC4C5', '#A4D5D6', '#CBE7E7'],
        "purple8": ['#2e1c40', '#482c63', '#623c86', '#7c4ca9', '#956bbd', '#ae8fcc', '#c8b2dc', '#e1d5ec'],
        "ctrl8": ['#513f06', '#816309', '#b1880c', '#e0ad0f', '#f1c232', '#f4d061', '#f8de91', '#fbedc1']
        }

    ataxia_score_max = { 
                     'Hindlimb dragging': 4, 
                     'Hindlimb splaying': 4,
                     'Forelimb dragging': 4, 
                     'Forelimb splaying': 4,
                     'Wobbling': 4,
                     'Nose down': 3, 
                     'Belly drag': 3,
                     }
    ataxia_explanations = {
                     'Hindlimb dragging': ['No drag', 'Limited motion', 'Occasional drag', 'Constant drag'], 
                     'Hindlimb splaying': ['No splaying', 'Brief splaying', 'Repeated splaying', 'Constant splaying'],
                    'Forelimb dragging': ['No drag', 'Limited motion', 'Occasional drag', 'Constant drag'], 
                    'Forelimb splaying': ['No splaying', 'Brief splaying', 'Repeated splaying', 'Constant splaying'],
                    'Wobbling': ['No wobbles', 'Wobbles', 'Wobbles & falls', 'Falls all the time'],
                     'Nose down': ['Normal', 'Nose lower', 'Nose down'], 
                     'Belly drag': ['No drag', 'Lower posture', 'Belly drag'], 
                           }
    
    subplot_labels = list(string.ascii_uppercase)

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
    # gives control over color (first item in each element of the provided orig_handle)
    # gives control over linestyle (second item in each element of the provided orig_handle)
    # there can be up to 4 such paired elements
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

class AnyObjectScatterer(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        s1 = plt.Line2D([0], [2], color = 'w', marker = 'o',markerfacecolor=orig_handle[0])
        returnable = [s1]
        if len(orig_handle)>1:
            s2 = plt.Line2D([0.6*width], [2], color = 'w', marker = 'o',markerfacecolor=orig_handle[1])
            returnable.append(s2)
        if len(orig_handle)>2:
            s3 = plt.Line2D([0], [2], color = 'w', marker = 'o',markerfacecolor=orig_handle[2])
            returnable.append(s3)
        if len(orig_handle)>3:
            s4 = plt.Line2D([0], [0], color = 'w', marker = 'o',markerfacecolor=orig_handle[3])
            returnable.append(s4)
        if len(orig_handle)>4:
            s5 = plt.Line2D([0], [0], color = 'w', marker = 'o',markerfacecolor=orig_handle[4])
            returnable.append(s5)
        if len(orig_handle)>4:
            s6 = plt.Line2D([0], [0], color = 'w', marker = 'o',markerfacecolor=orig_handle[5])
            returnable.append(s6)
        if len(orig_handle)>4:
            s7 = plt.Line2D([0], [0], color = 'w', marker = 'o',markerfacecolor=orig_handle[6])
            returnable.append(s7)
        return returnable

def hex_to_rgb(hex_clr):
    hex_clr = hex_clr.lstrip('#')
    return tuple(int(hex_clr[i:i+2], 16) / 255 for i in (0,2,4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def get_palette_from_html(html_clr, lightness_values):
    rgb = hex_to_rgb(html_clr)
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    palette = []
    for l in lightness_values:
        rgb_variant = colorsys.hls_to_rgb(hue, l, saturation)
        hex_variant = rgb_to_hex(rgb_variant)
        palette.append(hex_variant)
    return palette
