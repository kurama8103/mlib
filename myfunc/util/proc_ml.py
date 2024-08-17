import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def outliers(value, threshold=3):
    return np.abs(value) < (np.meaån(value) + threshold * np.std(value))


def calc_scores(value, predict=None):
    if type(value) == pd.DataFrame:
        value, predict, *_ = value.values.T.tolist()
    return {
        "R2": r2_score(value, predict),
        "MAE": mean_absolute_error(value, predict),
        "RMSE": np.sqrt(mean_squared_error(value, predict)),
        "MAPE": np.mean(np.abs((predict - value) / value)),
    }


def neutralize_series(series, by, proportion=1.0):
    # scores = series.values.reshape(-1, 1)
    # exposures = by.values.reshape(-1, 1)
    scores = series.reshape(-1, 1)
    exposures = by.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1))
    )
    correction = proportion * (exposures.dot(np.linalg.lstsq(exposures, scores)[0]))
    corrected_scores = scores - correction
    # neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    neutralized = corrected_scores.ravel()
    return neutralized


def unif(x):
    df = pd.Series(x)
    # return (scipy.stats.rankdata(x) - 0.5) / len(x)
    return (df.rank(method="first") - 0.5) / len(df)
