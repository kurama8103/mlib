import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def autocorr(df):
    result = np.correlate(df, df, mode='full')
    return result[int(result.size/2):]


def z_normalize(df):
    return (df-df.mean())/df.std()


def summary_plot(df, WINDOW=20):
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(12, 9))
    roll = df.rolling(WINDOW)
    roll.mean().plot(ax=ax[0, 0], title='mean', grid=True)
    roll.var().plot(ax=ax[0, 1], title='var', legend=False, grid=True)
    roll.skew().plot(ax=ax[1, 0], title='skew', legend=False, grid=True)
    roll.kurt().plot(ax=ax[1, 1], title='kurt', legend=False, grid=True)
    ax[1, 1].axhline(0, color="gray")


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


def geometric_brownian_motion(S0=100, T=1.0, r=0.05, sigma=0.2, I=100):
    # I:Number of trials
    z = np.random.standard_normal(I)
    ST = S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*z)
    return ST.mean()


def plot121(x1, x2):
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax1.plot(x1)
    ax2 = fig.add_subplot(122)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax2.plot(x2)
