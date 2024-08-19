import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


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
    correction = proportion * (
        exposures.dot(np.linalg.lstsq(exposures, scores, rcond=None)[0])
    )
    corrected_scores = scores - correction
    # neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    neutralized = corrected_scores.ravel()
    return neutralized


def unif(x):
    df = pd.Series(x)
    # return (scipy.stats.rankdata(x) - 0.5) / len(x)
    return (df.rank(method="first") - 0.5) / len(df)


def vis_features(X, y, figsize=(8, 12)):
    from yellowbrick.features import rank2d, pca_decomposition
    from sklearn.cluster import KMeans
    from yellowbrick.cluster.elbow import kelbow_visualizer

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize)

    rank2d(X, show=False, ax=axes[0])
    pca_decomposition(X, y, scale=True, proj_features=True, show=False, ax=axes[1])
    f = kelbow_visualizer(KMeans(), X, k=(2, 10), show=False, ax=axes[2])
    return fig, f.elbow_value_


def viz_model_regression(model, X_train, y_train, X_test, y_test):
    from yellowbrick.model_selection import feature_importances
    from yellowbrick.regressor import residuals_plot

    res = dict()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    _ = feature_importances(model, X_train, y_train, show=False, ax=axes[0])
    res["feat_imp"] = dict(zip(_.features_, _.feature_importances_))
    _ = residuals_plot(model, X_train, y_train, X_test, y_test, show=False, ax=axes[1])
    res["r2"] = {"train": _.train_score_, "test": _.test_score_}
    return res


from sklearn.preprocessing import KBinsDiscretizer


def categoricalize(x, n_bins=5):
    cls = KBinsDiscretizer(n_bins=n_bins, encode="ordinal")
    _ = cls.fit_transform(x.values.reshape(-1, 1)).reshape(-1).astype(int)
    return pd.Categorical(_), cls


def vis_model_classifier(model, X_train, y_train, X_test, y_test):
    import re
    from yellowbrick.classifier.rocauc import roc_auc
    from yellowbrick.classifier import precision_recall_curve, class_prediction_error
    from yellowbrick.classifier import classification_report, confusion_matrix

    pre = re.sub(r"[^A-Z]", "", str(model)) + "_"
    res = dict()
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 16))
    _ = roc_auc(
        model, X_train, y_train, X_test=X_test, y_test=y_test, show=False, ax=axes[0]
    )
    res[pre + "roc_auc"] = _.roc_auc
    _ = precision_recall_curve(
        model, X_train, y_train, X_test, y_test, per_class=True, show=False, ax=axes[1]
    )

    res[pre + "pr"] = _.score_
    class_prediction_error(
        model, X_train, y_train, X_test, y_test, show=False, ax=axes[2]
    )
    classification_report(
        model, X_train, y_train, X_test, y_test, support=True, show=False, ax=axes[3]
    )
    confusion_matrix(model, X_train, y_train, X_test, y_test, show=False, ax=axes[4])
    return res, fig
