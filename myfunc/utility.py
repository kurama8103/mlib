__version__ = '0.0.1'

import pandas as pd
from numpy import set_printoptions
import matplotlib.pyplot as plt
from seaborn import set_style
import japanize_matplotlib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def pref():
    #pd.options.display.max_rows = 100
    pd.options.display.max_columns = 100
    #pd.options.display.width = 120
    #pd.options.display.float_format = "{:,.4f}".format

    set_printoptions(suppress=True)
    # set_printoptions(precision=2)

    plt.figure()
    plt.rcParams['figure.figsize'] = 12, 4
    #rcParams['font.family']= 'Yu Mincho'

    set_style('whitegrid')
    japanize_matplotlib.japanize()

def end_of_month(df):
    if type(df) is not pd.DatetimeIndex:
        df = df.index
    df = df.sort_values()
    return df[(pd.Series(df.month.values).diff(-1) != 0).values]


def apply_concat(df, func, axis=0):
    return pd.concat([func(df[x].dropna()) for x in df.columns.values],
                     axis=axis, keys=df.columns.values)


def outliers(value, threshold=3):
    return np.abs(value) < (np.mean(value)+threshold*np.std(value))
    
    
def calc_scores(value, predict=None):
    if type(value) == pd.DataFrame:
        value, predict, *_ = value.values.T.tolist()
    return {
        'R2': r2_score(value, predict),
        'MAE': mean_absolute_error(value, predict),
        'RMSE': np.sqrt(mean_squared_error(value, predict)),
        'MAPE': np.mean(np.abs((predict - value) / value))
    }
