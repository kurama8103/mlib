import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd

#https://github.com/yhilpisch/py4fi/
def baysian_regression(x, y):
    
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=20)
        beta = pm.Normal('beta', mu=0, sd=20)
        sigma = pm.Uniform('sigma', lower=0, upper=50)

        y_est = alpha + beta * x
        likelihood = pm.Normal('x', mu=y_est, sd=sigma, observed=y)

        start = pm.find_MAP()
        step = pm.NUTS()
        trace = pm.sample(100, step, start=start, progressbar=False)

        fig = pm.traceplot(trace)
        plt.figure()

        return pd.DataFrame(
            [trace[n] for n in trace.varnames], index=trace.varnames).T
        
