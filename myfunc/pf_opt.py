import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import risk_models, expected_returns, plotting, black_litterman
from pypfopt.efficient_frontier import EfficientFrontier



def pf_opt(return_index):
    mu = expected_returns.mean_historical_return(return_index)
    s = risk_models.sample_cov(return_index)

    ef = EfficientFrontier(mu, s)
    ef.min_volatility()
    _r, _v, _sr = ef.portfolio_performance()
    res_opt = {
        'min_volatility': {
            'Expected annual return': _r,
            'annual volatility': _v,
            'Sharpe Ratio': _sr,
            'weight': ef.weights.tolist(),
        }}

    ef = EfficientFrontier(mu, s)
    ef.max_sharpe()
    _r, _v, _sr = ef.portfolio_performance()
    res_opt_ = {
        'max_sharpe': {
            'Expected annual return': _r,
            'annual volatility': _v,
            'Sharpe Ratio': _sr,
            'weight': ef.weights.tolist(),
        }}
    res_opt.update(res_opt_)

    ef = EfficientFrontier(mu, s, weight_bounds=(-1, 1))
    ef.max_sharpe()
    _r, _v, _sr = ef.portfolio_performance()
    res_opt_ = {
        'max_sharpe_short': {
            'Expected annual return': _r,
            'annual volatility': _v,
            'Sharpe Ratio': _sr,
            'weight': ef.weights.tolist(),
        }}
    res_opt.update(res_opt_)

    return res_opt


def pf_opt_plot(df, res_opt):
    mu = expected_returns.mean_historical_return(df)
    s = risk_models.sample_cov(df)

    ef = EfficientFrontier(mu, s)
    fig, ax = plt.subplots()
    plotting.plot_efficient_frontier(
        ef,
        ax=ax,
        weight_bounds=(None, None),
        show_assets=True,
        show_tickers=True
    )

#     for k, v in pd.DataFrame(res_opt).T[['Expected annual return', 'annual volatility']].iterrows():
#         ax.scatter(v[1], v[0], marker="*", label=k)

    [ax.annotate(i, (s[i][i]**(1/2)+0.002, mu[i])) for i in mu.index]
        
    pd.DataFrame({k: v['weight'] for k, v in res_opt.items()}).plot.bar()
    return plt.plot()
