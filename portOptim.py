import os
from functools import partial
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from woa import WhaleOptim


class PortfolioOptimizer:

    def __init__(self, price, lb, ub, n_iter):
        self.price = price
        self.lb = lb
        self.ub = ub
        self.n_iter = n_iter
        os.makedirs('result', exist_ok=True)


    def init_param(self):

        self.rtn = np.log(self.price / self.price.shift(1)).fillna(0).values
        self.mean_rtn = self.rtn.mean(axis=0)
        self.cov_rtn = np.cov(self.rtn, rowvar=False)

    def obj_func(self, weights, risk_aversion):
        port_rtn = np.dot(weights, self.mean_rtn) * 252
        if weights.ndim > 1:
            port_vol = np.diagonal(np.dot(np.dot(weights, self.cov_rtn), weights.T)) * 252
        else:
            port_vol = np.dot(np.dot(weights, self.cov_rtn), weights.T) * 252
        sharpe = np.sqrt(port_vol) * risk_aversion - (1 - risk_aversion) * port_rtn
        return sharpe

    def woa_minimize(self, n_whale, n_iter, spiral_constant=0.5):
        weights = []
        for a in tqdm(np.linspace(0, 1, self.n_iter)):
            f = partial(self.obj_func, risk_aversion=a)
            woa = WhaleOptim(f, n_whale, spiral_constant, n_iter, self.lb, self.ub)
            _, optimal_x = woa.run()
            weights.append(optimal_x)
        weights = np.array(weights)
        np.save('result/weights_woa.npy', weights)
        return weights

    def scipy_optimizer(self, risk_aversion):

        def constraint_sum(weights):
            return weights.sum() - 1

        x0 = np.ones(self.mean_rtn.shape[-1]) / self.mean_rtn.shape[-1]
        constraints = ({'type': 'eq', 'fun': constraint_sum})
        bounds = list(zip(self.lb, self.ub))
        optim = optimize.minimize(self.obj_func, x0=x0, args=(risk_aversion,),
                                  bounds=bounds, constraints=constraints)
        return optim

    def scipy_minimize(self):
        weights = []
        for a in tqdm(np.linspace(0, 1, self.n_iter)):
            result = self.scipy_optimizer(risk_aversion=a)
            weights.append(result.x)
        weights = np.array(weights)
        np.save('result/weights_scipy.npy', weights)
        return weights

    def calc_perf(self, weights):
        port_rtn = np.dot(weights, self.mean_rtn) * 252
        port_vol = np.diagonal(np.dot(np.dot(weights, self.cov_rtn), weights.T)) * 252
        sharpe = port_rtn / np.sqrt(port_vol)
        return port_rtn, port_vol, sharpe

    def visualize(self, port_rtn, port_vol, sharpe, name=''):
        plt.figure(figsize=(15, 8))
        plt.scatter(port_vol, port_rtn, c=sharpe)
        plt.colorbar()
        plt.scatter(port_vol[sharpe.argmax()], port_rtn[sharpe.argmax()],
                    marker='o', s=500, color='blue')
        plt.scatter(port_vol[port_vol.argmin()], port_rtn[port_vol.argmin()],
                    marker='o', s=500, color='green')
        plt.savefig(f"result/efficient_frontier_{name}")
        plt.show()

