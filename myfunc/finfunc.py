import scipy.fftpack as sf
import scipy.signal as sg
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
plt.style.use('seaborn')

'''def autocorr(df):
    result = np.correlate(df, df, mode='full')
    return result[int(result.size/2):]'''


def z_normalize(df):
    return (df-df.mean())/df.std()


def rolling_mvsk(df, WINDOW=20):
    r = df.rolling(WINDOW)
    df = pd.concat([df, r.mean(), r.var(), r.skew(), r.kurt()], axis=1)
    df.columns = ['value', 'mean', 'var', 'skew', 'kurt']
    return df


def summary_plot(df, WINDOW=20):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(12, 9), gridspec_kw=dict(hspace=0.2))
    fig.suptitle('summary plot : window='+str(WINDOW),fontsize=15)
    ax[0].plot(df[WINDOW:],color='gray') 
    
    df = rolling_mvsk(df=df, WINDOW=20)
    ax[0].plot(df['mean'])
    ax[0].set_title('mean')
    ax[1].plot(df['var'])
    ax[1].set_title('var')
    ax[2].plot(df['skew'])
    ax[2].set_title('skew')
    ax[3].plot(df['kurt'])
    ax[3].set_title('kurt')
    #ax[3].axhline(0, color="gray")
    #ax[0, 0].grid(True)


def Ornstein_Uhlenbeck_process(x_initial=80, x_drift=100, sigma=0.1):
    tau = 0.05  # Time constant
    dt = 1.0/365
    T = 1.0
    n = int(T/dt)  # Number of time steps
    t = np.linspace(0., T, n)  # Time vector

    sigma_bis = sigma*np.sqrt(2./tau)
    sqrtdt = np.sqrt(dt)

    x = np.zeros(n)
    x[0] = x_initial
    for i in range(n-1):
        x[i+1] = (x[i]
                  + dt*(-(x[i]-x_drift)/tau)
                  + sigma_bis*sqrtdt*np.random.randn())

    #plt.plot(t, df)
    plt.plot(t, x)


def Ornstein_Uhlenbeck_process_log(x_initial=80, x_drift=100, sigma=0.1, half_life=0.5):
    dt = 1.0/365
    T = 1.0
    n = int(T/dt)  # Number of time steps
    t = np.linspace(0., T, n)  # Time vector

    eta = np.log(2)/half_life  # Mean-reversion factor
    eta_f = np.exp(-eta*dt)
    x = np.zeros(n)
    x[0] = x_initial
    for i in range(n-1):
        x[i+1] = np.exp(np.log(x[i])*eta_f
                        + np.log(x_drift)*(1-eta_f)
                        - (1-eta_f**2)*(sigma**2)/(4*eta)
                        + sigma*np.random.randn()*np.sqrt((1-eta_f**2)/2*eta))

    plt.plot(t, x)
    # return x


def heston_stochastic_volatility_model(
        x_initial=100, v_initial=0.1, v_drift=0.1, v_vol=0.1, v_reversion=3.0, r=0.05, rho=0.6):
    #S0 = 100.
    #r = 0.05
    # v0 = 0.1 #Initial volatility
    # kappa = 3.0 #Mean-reversion factor
    # theta = 0.25 #Volatility equilibrium
    # sigma = 0.1 #vvol
    # rho = 0.6 #Correlation of dS and dv

    # Cholesky decomposition of the correlation matrix
    corr_mat = np.zeros((2, 2))
    corr_mat[0, :] = [1.0, rho]
    corr_mat[1, :] = [rho, 1.0]
    cho_mat = np.linalg.cholesky(corr_mat)

    T = 1.0
    M = 50
    I = 100
    ran_num = np.random.standard_normal((2, M + 1, I))

    # volatility process
    dt = T / M
    v = np.zeros_like(ran_num[0])
    vh = np.zeros_like(v)
    v[0] = v_initial
    vh[0] = v_initial
    for t in range(1, M + 1):
        ran = np.dot(cho_mat, ran_num[:, t, :])
        vh[t] = (vh[t - 1] + v_reversion * (v_drift - np.maximum(vh[t - 1], 0)) * dt
                 + v_vol * np.sqrt(np.maximum(vh[t - 1], 0)) * np.sqrt(dt)
                 * ran[1])
    v = np.maximum(vh, 0)

    # index level process,
    S = np.zeros_like(ran_num[0])
    S[0] = x_initial
    for t in range(1, M + 1):
        ran = np.dot(cho_mat, ran_num[:, t, :])
        S[t] = S[t-1] * np.exp((r - 0.5 * v[t]) * dt +
                               np.sqrt(v[t]) * ran[0] * np.sqrt(dt))

    plt.plot(np.mean(S, axis=1))
    plt.plot(S[:, :10])
    return np.mean(S, axis=1)


def geometric_brownian_motion(S0=100, T=1.0, r=0.05, sigma=0.2, I=100):
    # I:Number of trials
    z = np.random.standard_normal(I)
    ST = S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*z)
    return ST.mean()


'''def plot121(x1, x2):
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot(121)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax1.plot(x1)
    ax2 = fig.add_subplot(122)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax2.plot(x2)'''


def acf_plot(ts):
    fig, ax = plt.subplots(3, 1, sharex=False, figsize=(
        12, 9), gridspec_kw=dict(hspace=0.5))
    ax[0].plot(ts)
    # ax[0].set_title('Residual')
    fig = sm.graphics.tsa.plot_acf(ts, lags=30, ax=ax[1])
    fig = sm.graphics.tsa.plot_pacf(ts, lags=30, ax=ax[2])
    fig.suptitle('Residual')


def adf_summary(ts):
    tsa = sm.tsa.adfuller(ts, regression='nc')
    return {'adf': str(tsa[0]),
            'pvalue': str(tsa[1]),
            'usedlag': str(tsa[2]),
            'nobs': str(tsa[3]),
            'critical values': str(tsa[4]),
            'icbest': str(tsa[5])}


def HistgramAndPDF(price, WINDOW=1):
    from scipy.stats import norm
    rn = price.pct_change(WINDOW).dropna()
    x = np.linspace(float(rn.min()), float(rn.max()), 100)
    pdf = norm.pdf(x, rn.mean(), rn.std())

    # plot
    fig = plt.figure(figsize=(6, 3))
    ax = plt.subplot(1, 1, 1)
    rn.hist(bins=100, density=True, ax=ax)
    plt.plot(x, pdf)
    plt.xlabel('$P_{t}/P_{t-'+str(WINDOW)+'}-1$')
    plt.ylabel('probability density function')

    print("mean %2.5f std %2.5f skew %2.5f kurt %2.5f"
          % (rn.mean(), rn.std(), rn.skew(), rn.kurt()))
    # kurt base: 3


def statmodels_summary(model_fit, period=5):
    date_future_start = model_fit.resid.index[-1]
    date_future_end = model_fit.resid.index[-1] + \
        (date_future_start-model_fit.resid.index[-period])
    plt.figure(figsize=(12, 3))
    plt.plot(model_fit.resid + model_fit.predict(), label='value')
    plt.plot(model_fit.predict(
        model_fit.resid.index[0], date_future_end), "r", label='predict')
    plt.title('Predict')
    plt.legend()

    acf_plot(model_fit.resid)
    return model_fit.summary()


def hurst(ts):
    # hurst Exponent
    '''
    H<0.5 : 時系列データが平均回帰性を持つ
    H=0.5 : 時系列データが幾何ブラウン運動である
    H>0.5 : 時系列データがトレンドを持つ
    '''
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(np.array(ts[lag:]), np.array(ts[:-lag]))))
           for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2


def fft_plot(df):
    temp_fft = sf.fft(df)
    temp_psd = np.abs(temp_fft)**2
    fftfreq = sf.fftfreq(len(temp_psd), 1. / 365)

    # PSD
    i = fftfreq > 0  # Only positive freq
    plt.figure(figsize=(12, 3))
    plt.plot(fftfreq[i], 10*np.log10(temp_psd[i]))  # Log, dB
    plt.xlabel('Frequency/year')
    plt.ylabel('PSD(dB)')

    # FFT
    temp_fft_bis = temp_fft.copy()
    temp_fft_bis[np.abs(fftfreq) > 5] = 0  # Cut high freq
    temp_slow = np.real(sf.ifft(temp_fft_bis))
    plt.figure(figsize=(12, 3))
    plt.plot_date(df.index, df, '-', lw=.5)
    plt.plot_date(df.index, temp_slow, '-')


def filter_FIR_plot(df, window=2):
    h = sg.get_window('triang', window)
    fil = sg.convolve(df, h/h.sum())
    fil[:window] = None
    plt.figure(figsize=(12, 3))
    plt.plot(df, lw=.5)
    plt.plot(df.index, fil[:len(df)])
    plt.title('Finite Impulse Response Filter  window=' + str(window))


def filter_IIR_Butterworth_plot(df, dimention=1):
    # low
    b, a = sg.butter(dimention, 2./365, btype='low')
    plt.figure(figsize=(12, 3))
    plt.plot(df.index, df, lw=.5, label='value')
    plt.plot(df.index, sg.filtfilt(b, a, df), label='low')
    plt.legend()
    plt.title('Infinite Impulse Response Filter  dimention=' + str(dimention))

    # high
    d, c = sg.butter(dimention, 2./365, btype='high')
    plt.figure(figsize=(12, 3))
    plt.plot(df.index, df, lw=.5, label='value')
    plt.plot(df.index, sg.filtfilt(d, c, df), label='high')
    plt.legend()


import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix

class quick_LightGBM:
    def __init__(self,train,target,params):
        self.train=train
        self.target=target
        self.params=params
        self.train_lgb = lgb.Dataset(self.train,self.target)
        #validation_data = lgb.Dataset(df_test, reference=train_data)

    def train_model(self):
        self.model = lgb.train(    
            self.params,
            self.train_lgb,
            num_boost_round=100,
            verbose_eval=10,
            #early_stopping_rounds=5,
            #valid_sets=validation_data,
            )

        self._pred=self.model.predict(self.train,num_iteration=self.model.best_iteration)
        self.pred_max = np.argmax(self._pred, axis=1) 
    
    def confusion_matrix(self):
        return confusion_matrix(self.pred_max,self.target)

    def plot_importance(self):
        return lgb.plot_importance(self.model)

    def feature_importance(self):
        return pd.DataFrame({
            'name':self.train.columns,
            'value':np.array(self.model.feature_importance())
            })
    
def getEndOfMonth(df):
    df=df.sort_index()
    return df[(pd.Series(df.index.month.values).diff(-1) != 0).values]

def high_water_mark(price_index, window=None):
    if window is None:
        window=len(price_index)
    return price_index.rolling(window, min_periods=1).max()


def max_draw_down(price_index, window=None):
    if window is None:
        window=len(price_index)
    return price_index/high_water_mark(price_index, window)-1


def decompose(return_index:pd.Series,period=20)->pd.DataFrame:
        dcp=sm.tsa.seasonal_decompose(
            return_index.dropna(), period=period,two_sided=False)
        return pd.DataFrame({
            'dcp_trend':dcp.trend,
            'dcp_seasonal':dcp.seasonal,
            'dcp_resid':dcp.resid,
        })

def acfs(return_pct, nlags=10):
    return pd.concat([
        return_pct.apply(lambda x: sm.tsa.stattools.acf(x.dropna(), nlags=nlags)),
        return_pct.apply(lambda x: sm.tsa.stattools.pacf(x.dropna(), nlags=nlags, 
                                                         method='ols'))], 
        axis=0, keys=['acf', 'pacf'])
        