import matplotlib.pyplot as plt

DPI = 144

_params = {'text.usetex' : True,
          'text.latex.preamble':[r"\usepackage{lmodern}", r"\usepackage{amsmath}"],
          'font.size' : 8,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          'lines.linewidth': 0.5,
          'axes.titlesize': 'large',
          'axes.linewidth': 0.3,
          'xtick.major.width': 0.25,
          'ytick.major.width': 0.25,
          'grid.linewidth': 0.125,
          'grid.linestyle': ':',
          'grid.color': [0.5, 0.5, 0.5],
          'legend.borderaxespad': 0.1,
          'legend.fontsize': 8,
          'legend.frameon': True,
          'xtick.minor.size': 1,
          'ytick.minor.size': 1,
          'xtick.major.size': 2,
          'ytick.major.size': 2,
          'figure.autolayout':True,
          'savefig.dpi': 300,
          'savefig.bbox': 'tight',
          'savefig.pad_inches': 0.1
          }

def latex_preamble_append(*args):
    plt.rcParams['text.latex.preamble'].append(*args)

def setup(param_updates=dict()):
    params = dict()
    params.update(_params)
    params.update(param_updates)
    plt.rcParams.update(params)

def paper_figure(width_pt, height_pt):
    # simple funtion to get a figure with correct size (given a w,h in points)
    # by default matplotlib adds a margin of 0.1 inches. when setting int
    # smaller than that the plot is not resized, but cropped, and thus changing
    # the figure size. Therefore we add 2*0.1 on each size + 2* the margin that
    # we set (0.01). We have to set something > 0, because it crops some of the
    # text otherwise.
    # 72 is the standard dpi that we set
    return plt.figure(figsize=((width_pt / 72.) + 0.22, (height_pt / 72.) + 0.22), dpi=144)



