import numpy as np
import pandas as pd
from portOptim import PortfolioOptimizer


if __name__ == '__main__':
    stock_df = pd.read_csv('data/sample_portfolio.csv', parse_dates=['Date'], index_col=['Date'])
    print('종목들: \n', stock_df.columns.tolist())

    lb = np.repeat(0.02, 15)
    ub = np.repeat(0.2, 15)
    n_iter = 500

    woa = True

    optim = PortfolioOptimizer(stock_df, lb, ub, n_iter)
    optim.init_param()
    if woa:
        weights = optim.woa_minimize(n_whale=500, n_iter=500, spiral_constant=0.5)
    else:
        weights = optim.scipy_minimize()
    port_rtn, port_vol, sharpe = optim.calc_perf(weights)
    name = 'woa' if woa else 'scipy'
    optim.visualize(port_rtn, port_vol, sharpe, name=name)


