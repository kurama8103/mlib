import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as sf
import scipy.signal as sg
import scipy.stats as scs
import statsmodels.api as sm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def z_normalize(x):
    return (x-x.mean())/x.std()


def rolling_mvsk(df: pd.Series, window=20):
    r = df.rolling(window)
    df = pd.concat([df, r.mean(), r.var(), r.skew(), r.kurt()], axis=1)
    df.columns = ['value', 'mean', 'var', 'skew', 'kurt']
    return df


def summary_plot(df: pd.Series, window=20):
    fig, ax = plt.subplots(4, 1, sharex=True, gridspec_kw=dict(hspace=0.2))
    fig.suptitle('summary plot : window='+str(window), fontsize=15)
    ax[0].plot(df[window:], color='gray')

    df = rolling_mvsk(df=df, window=20)
    ax[0].plot(df['mean'])
    ax[0].set_title('mean')
    ax[1].plot(df['var'])
    ax[1].set_title('var')
    ax[2].plot(df['skew'])
    ax[2].set_title('skew')
    ax[3].plot(df['kurt'])
    ax[3].set_title('kurt')


def acfs(x, nlags=10):
    return pd.DataFrame({
        'acf': sm.tsa.stattools.acf(x, nlags=nlags),
        'pacf': sm.tsa.stattools.pacf(x, nlags=nlags, method='ols')
    })


def acf_plot(x, lags=30):
    return sm.graphics.tsa.plot_acf(x, lags=lags), sm.graphics.tsa.plot_pacf(x, lags=lags)


def adf_summary(x):
    tsa = sm.tsa.adfuller(x, regression='nc')
    return {
        'adf': tsa[0],
        'pvalue': tsa[1],
        'usedlag': tsa[2],
        'nobs': tsa[3],
        'critical values': tsa[4],
        'icbest': tsa[5]
    }


def hurst(x, dim=100):
    '''
    H<0.5 : anti-trend (mean-regression)
    H=0.5 : brownian (martingale)
    H>0.5 : trend
    '''
    dim = min(100, len(x)-2)
    if dim > 3:
        lags = range(2, dim)
        tau = [np.sqrt(np.std(np.subtract(np.array(x[lag:]), np.array(x[:-lag]))))
               for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2


def high_water_mark(return_index: pd.Series, window=None):
    if window is None:
        window = len(return_index)
    return return_index.rolling(window, min_periods=1).max()


def max_draw_down(return_index: pd.Series, window=None):
    if window is None:
        window = len(return_index)
    return return_index/high_water_mark(return_index, window)-1


def decompose(return_index, period=20) -> pd.DataFrame:
    dcp = sm.tsa.seasonal_decompose(
        return_index, period=period, two_sided=False)
    return pd.DataFrame({
        'dcp_trend': dcp.trend,
        'dcp_seasonal': dcp.seasonal,
        'dcp_resid': dcp.resid,
    })


def value_at_risk(x, q=[0.01, 1, 5, 10]):
    return pd.Series(scs.scoreatpercentile(x, q),
                     index=q, name='value_at_risk')


def normality_tests(x):
    return {
        'skew': scs.skew(x),
        'skew_pvalue': scs.skewtest(x)[1],
        'kurtosis': scs.kurtosis(x),
        'kurtosis_pvalue': scs.kurtosistest(x)[1],
        'normal_pvalue': scs.normaltest(x)[1],
    }


def cvar(x, q=[0.01, 1, 5, 10]):
    var = value_at_risk(x, q)
    return pd.Series([x[x <= var[i]].mean() for i in [0.01, 1, 5, 10]],
                     index=q, name='cvar')


def histgram_pdf(x, bins=50):
    plt.hist(x, bins=bins, density=True)
    plt.plot(probability_density_function(x))
    plt.title('probability density function')


def probability_density_function(x):
    n = np.linspace(float(x.min()), float(x.max()), 100)
    return pd.Series(scs.norm.pdf(n, x.mean(), x.std()), index=n)


def fast_Fourier_transform_psd(x):
    temp_fft = sf.fft(x)
    temp_psd = np.abs(temp_fft)**2
    fftfreq = sf.fftfreq(len(temp_psd), 1. / 365)

    i = fftfreq > 0  # Only positive freq
    psd = pd.Series(10*np.log10(temp_psd[i]),
                    name='PSD_dB', index=fftfreq[i])
    psd.index.name = 'freq/y'
    return psd


def fast_Fourier_transform(df: pd.Series, freq_cut=5):
    temp_fft = sf.fft(df.values)
    temp_psd = np.abs(temp_fft)**2
    fftfreq = sf.fftfreq(len(temp_psd), 1. / 365)

    temp_fft_bis = temp_fft.copy()
    temp_fft_bis[np.abs(fftfreq) > freq_cut] = 0  # Cut high freq
    temp_slow = np.real(sf.ifft(temp_fft_bis))
    return pd.Series(temp_slow,
                     name='FFT_freq<='+str(freq_cut), index=df.index)


def fast_Fourier_transform_(nda: np.ndarray, freq_cut=5):
    temp_fft = sf.fft(nda)
    temp_psd = np.abs(temp_fft)**2
    fftfreq = sf.fftfreq(len(temp_psd), 1. / 365)

    temp_fft_bis = temp_fft.copy()
    temp_fft_bis[np.abs(fftfreq) > freq_cut] = 0  # Cut high freq
    temp_slow = np.real(sf.ifft(temp_fft_bis))
    return {'FFT_freq<='+str(freq_cut):temp_slow}


def filter_finite_impulse_response(df: pd.Series, window=2):
    h = sg.get_window('triang', window)
    fil = sg.convolve(df, h/h.sum())
    fil[:window] = None
    return pd.Series(fil[:len(df)], name='FIR_'+str(window),
                     index=df.index)


def filter_IIR_Butterworth(df: pd.Series, dimension=1):
    b, a = sg.butter(dimension, 2./365, btype='low')
    d, c = sg.butter(dimension, 2./365, btype='high')
    return pd.DataFrame({
        'IIR_low_dim'+str(dimension): sg.filtfilt(b, a, df),
        'IIR_high_dim'+str(dimension): sg.filtfilt(d, c, df)
    }, index=df.index)


def statmodels_predict(model_fit, period=5):
    pred_start = model_fit.resid.index[-1]
    pred_end = pred_start + (pred_start-model_fit.resid.index[-period])

    pred = pd.DataFrame({
        'value': model_fit.resid + model_fit.predict(),
        'predict': model_fit.predict(model_fit.resid.index[0], pred_end)
    })
    pred.index = pd.to_datetime(pred.index)
    return pred


def sklearn_predict_regression(model, list_feature_target, n_round=2):
    features_train, features_test, target_train, target_test = list_feature_target
    test = pd.concat([target_train, target_test])
    features_model = model.fit(features_train, target_train)
    predict = features_model.predict(features_test)

    try:
        coef = model.coef_.round(n_round)
        intercept = model.intercept_.round(n_round)
    except:
        coef = None
        intercept = None

    print(pd.Series({
        'features_model': features_model,
        'score': model.score(features_test, target_test).round(n_round),
        'n_features_original': features_train.shape[1],
        'coef': coef,
        'intercept': intercept,
    }))

    return pd.DataFrame({
        'value': test,
        'predict': np.concatenate([features_model.predict(features_train), predict])
    }, index=test.index)


def dynamic_time_warping(x, y):
    return fastdtw(x, y, dist=euclidean)
