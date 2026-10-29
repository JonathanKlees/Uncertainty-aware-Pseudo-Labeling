import os
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter, FuncFormatter

from matplotlib.colors import LogNorm, Normalize
from matplotlib.colors import ListedColormap

# Add the LaTeX bin directory to the PATH
def set_plot_layout(path_to_latex):
    os.environ['PATH'] = os.path.expanduser(path_to_latex) + os.pathsep + os.environ['PATH']
    plt.rc('text', usetex=True)
    plt.rc('font', family='Computer Modern Roman')

    matplotlib.rcParams['lines.linewidth'] = 1
    matplotlib.rcParams['axes.labelpad'] = 10
    plt.rcParams['font.sans-serif'] = "cmr10"
    plt.rcParams['font.size'] = 10.5

    matplotlib.rcParams['axes.titlesize'] = 10.5
    matplotlib.rcParams['axes.labelsize'] = 10.5
    matplotlib.rcParams['xtick.labelsize'] = 10.5
    matplotlib.rcParams['ytick.labelsize'] = 10.5

    matplotlib.rcParams["axes.prop_cycle"] = cycler('color', ["teal", "coral", "palevioletred", "slategrey", "forestgreen",  "darkmagenta",  "gold", "steelblue", "bisque", "darkseagreen"])
    matplotlib.rcParams['legend.framealpha'] = 1
    matplotlib.rcParams['figure.figsize']=[5.8, (9/16)*5.8]
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['legend.fontsize'] = 9
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['axes.edgecolor'] = "slategray"
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['figure.titlesize'] = 'medium'

    plot_params = dict()
    plot_params["quadratic_image_size"] = (3.2625, 3.2625)
    plot_params["large_image_size"] = (5.8, 6.525)
    plot_params["large_quadratic_image_size"] = (5.8, 5.8)

    plot_params["small_ms"] = 1
    plot_params["large_ms"] = 2
    plot_params["huge_annot_size"] = 25

    plot_params["colors"] = ["teal", "coral", "palevioletred", "slategrey", "forestgreen",  "darkmagenta",  "gold", "steelblue", "bisque", "darkseagreen"]

    return plot_params