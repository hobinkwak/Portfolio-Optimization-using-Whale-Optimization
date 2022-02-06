## Portfolio Optimization using Whale Optimization

### Overview
- simple implementation for portfolio optimization using whale optimization algorithm
- compare results between whale optimization and scipy.optimize
- draw efficient frontier from varying risk aversion value with each optimizing algorithm


### Data
- 15 stocks in DJIA from 2014 to 2019
  - AAPL, AXP, BA, CAT, CSCO, DIS, GS, HD, IBM, JPM, KO, MCD, MRK, UNH, WBA
- Daily Adj Close price data

### Objective Function
- maximize portfolio return & minimize portfolio volatility
  - with risk aversion param \[0, 1]
 
```python
def obj_func(weights, mean_rtn, cov_rtn, risk_aversion):
    port_rtn = np.dot(weights, mean_rtn) * 252
    port_vol = np.diagonal(np.dot(np.dot(weights, cov_rtn), weights.T)) * 252
    sharpe = np.sqrt(port_vol) * risk_aversion - (1 - risk_aversion) * port_rtn
    return sharpe 
``


