__version__ = '0.0.1'

from pandas import options
from numpy import set_printoptions
from matplotlib.pylab import rcParams
from seaborn import set_style

def pref():
    #options.display.max_rows = 100
    options.display.max_columns = 100
    #options.display.width = 120
    #options.display.float_format = "{:,.4f}".format
    set_printoptions(suppress=True)
    #set_printoptions(precision=2)
    rcParams['figure.figsize'] = 12, 6
    #rcParams['font.family']= 'Yu Mincho'
    set_style('darkgrid')


