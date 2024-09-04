import numpy as np
import pandas as pd


class TimeSeries:
    def __init__(self, data: pd.DataFrame, log=False):
        if data.min().min() > 0:
            self.price = data
            self.return_pct_raw = data.pct_change()
        else:
            self.return_pct_raw = data
            self.price = None

        self.return_index = (self.return_pct_raw+1).cumprod().asfreq('B')
        self.return_pct = (self.return_index.apply(np.log)).diff(
        ) if log is True else self.return_index.pct_change()
        self.return_pct_d = self.return_pct.dropna()
        self.return_index_inv = (1-self.return_pct).cumprod()
