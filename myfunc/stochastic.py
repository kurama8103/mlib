import numpy as np
import scipy.stats as scs


def geometric_brownian_motion(S0=100, T=1.0, M=250, r=0.03, sigma=0.2, npath=100):
    dt = T / M
    S = np.zeros((M + 1, npath))
    S[0] = S0
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * np.random.standard_normal(npath)
        )
    return S


def ornstein_uhlenbeck_process(
    x_initial=80, x_drift=100, tau=0.1, T=1.0, M=250, sigma=0.2, npath=100
):
    """
    tau : Time constant
    """

    dt = T / M
    sigma_bis = sigma * np.sqrt(2.0 / tau)

    x = np.zeros((M + 1, npath))
    x[0] = x_initial
    for i in range(M):
        x[i + 1] = (
            x[i]
            + dt * (-(x[i] - x_drift) / tau)
            + sigma_bis * np.sqrt(dt) * np.random.standard_normal(npath)
        )
    return x


def ornstein_uhlenbeck_process_log(
    x_initial=80, x_drift=100, half_life=0.5, T=1.0, M=250, sigma=0.2, npath=100
):
    dt = T / M
    eta = np.log(2) / half_life  # Mean-reversion factor
    eta_f = np.exp(-eta * dt)
    x = np.zeros((M + 1, npath))
    x[0] = x_initial
    for i in range(M):
        x[i + 1] = np.exp(
            np.log(x[i]) * eta_f
            + np.log(x_drift) * (1 - eta_f)
            - (1 - eta_f**2) * (sigma**2) / (4 * eta)
            + sigma
            * np.random.standard_normal(npath)
            * np.sqrt((1 - eta_f**2) / 2 * eta)
        )
    return x


def heston_stochastic_volatility_model(
    x_initial=100,
    v_initial=0.1,
    v_drift=0.2,
    v_vol=0.1,
    v_reversion=3.0,
    rho=0.6,
    T=1.0,
    M=250,
    r=0.05,
    npath=100,
):
    """
    v0 : Initial volatility
    kappa : Mean-reversion factor
    theta : Volatility equilibrium
    sigma : vvol
    rho : Correlation of dS and dv
    """

    # Cholesky decomposition of the correlation matrix
    cho_mat = cholesky_decomposition(rho)
    ran_num = np.random.standard_normal((2, M + 1, npath))

    # volatility process
    dt = T / M
    v = np.zeros_like(ran_num[0])
    vh = np.zeros_like(v)
    v[0] = v_initial
    vh[0] = v_initial
    for t in range(1, M + 1):
        ran = np.dot(cho_mat, ran_num[:, t, :])
        vh[t] = (
            vh[t - 1]
            + v_reversion * (v_drift - np.maximum(vh[t - 1], 0)) * dt
            + v_vol * np.sqrt(np.maximum(vh[t - 1], 0)) * np.sqrt(dt) * ran[1]
        )
    v = np.maximum(vh, 0)

    # index level process,
    S = np.zeros_like(ran_num[0])
    S[0] = x_initial
    for t in range(1, M + 1):
        ran = np.dot(cho_mat, ran_num[:, t, :])
        S[t] = S[t - 1] * np.exp(
            (r - 0.5 * v[t]) * dt + np.sqrt(v[t]) * ran[0] * np.sqrt(dt)
        )
    return S


def cholesky_decomposition(rho):
    corr_mat = np.zeros((2, 2))
    corr_mat[0, :] = [1.0, rho]
    corr_mat[1, :] = [rho, 1.0]
    return np.linalg.cholesky(corr_mat)


from math import log, sqrt, exp
from scipy import stats


# Analytical Black-Scholes-Merton (BSM) Formula
def bsm_call_value(S0=100, K=100, T=1.0, r=0.0, sigma=0.2):
    """Valuation of European call option in BSM model.
    Analytical formula.

    Parameters
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity date (in year fractions)
    r : float
        constant risk-free short rate
    sigma : float
        volatility factor in diffusion term

    Returns
    =======
    value : float
        present value of the European call option
    """

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    value = S0 * stats.norm.cdf(d1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf(
        d2, 0.0, 1.0
    )
    # stats.norm.cdf —> cumulative distribution function
    # for normal distribution
    return value


# Vega function
def bsm_vega(S0=100, K=100, T=1.0, r=0.0, sigma=0.2):
    """Vega of European option in BSM model.
    Parameters
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity date (in year fractions)
    r : float
        constant risk-free short rate
    sigma : float
        volatility factor in diffusion term

    Returns
    =======
    vega : float
        partial derivative of BSM formula with respect to sigma, i.e. Vega
    """

    from math import log, sqrt
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    vega = S0 * stats.norm.cdf(d1, 0.0, 1.0) * sqrt(T)
    return vega


# Implied volatility function
def bsm_call_imp_vol(S0=100, K=100, T=1.0, r=0.0, C0=0, sigma_est=0.2, it=100):
    """Implied volatility of European call option in BSM model.
    Parameters
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity date (in year fractions)
    r : float
        constant risk-free short rate
    sigma_est : float
        estimate of impl. volatility
    it : integer
        number of iterations

    Returns
    =======
    simga_est : float
        numerically estimated implied volatility"""

    for i in range(it):
        sigma_est -= (bsm_call_value(S0, K, T, r, sigma_est) - C0) / bsm_vega(
            S0, K, T, r, sigma_est
        )
    return sigma_est
