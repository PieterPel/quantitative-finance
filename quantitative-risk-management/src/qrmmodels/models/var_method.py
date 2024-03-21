from abc import ABC, abstractmethod
import math
import pandas as pd
import numpy as np
from scipy.stats import norm, kendalltau, t
from sklearn.mixture import GaussianMixture
import arch
from statsmodels.distributions.copula.api import ClaytonCopula
from pycop import archimedean

from ..utils import get_total_loss_array, check_for_bad_df, check_for_fitted_model, make_hill_plot, hill_estimate


class VarMethod(ABC):

    def __init__(self):
        self.fitted_model = None

    def fit(self, df: pd.DataFrame, pct_stock_1) -> None:
        check_for_bad_df(df)
        self._fit_without_check(df, pct_stock_1)

    def get_var(self, alpha: float) -> float:
        check_for_fitted_model(self)
        return self._get_var_without_check(alpha)

    @abstractmethod
    def _fit_without_check(self, df: pd.DataFrame, pct_stock_1: float) -> None:
        ...

    @abstractmethod
    def _get_var_without_check(self, alpha: float) -> float:
        ...


class VarCovarMethod(VarMethod):

    def _fit_without_check(self, df: pd.DataFrame, pct_stock_1) -> None:

        # First approach, probably not correct
        columns = df.columns

        mean_loss_1 = df[columns[0]].mean()
        mean_loss_2 = df[columns[1]].mean()

        std_loss_1 = df[columns[0]].std()
        std_loss_2 = df[columns[1]].std()

        corr = df[columns[0]].corr(df[columns[1]])
        covar = corr * std_loss_1 * std_loss_2

        mean = pct_stock_1 * mean_loss_1 + (1-pct_stock_1) * mean_loss_2
        var = (pct_stock_1 * std_loss_1) ** 2 + ((1-pct_stock_1) * std_loss_2) ** 2 + 2 * pct_stock_1 * (1-pct_stock_1) * covar
        std = math.sqrt(var)

        self.fitted_model = {"mean_total": mean,
                             "std_total": std,
                             "var_total": var,
                             "mean_1": mean_loss_1,
                             "mean_2": mean_loss_2,
                             "var_loss_1": std_loss_1 ** 2,
                             "var_loss_2": std_loss_2 ** 2,
                             "covar": covar}

    def _get_var_without_check(self, alpha: float) -> float:
        # VaR = norm.ppf(alpha, self.fitted_model['mean'], self.fitted_model['std'])
        mu = self.fitted_model['mean_total']
        std = self.fitted_model['std_total']

        VaR = mu + std * norm.ppf(alpha)

        return VaR


class HistoricalSimulationMethod(VarMethod):

    def _fit_without_check(self, df: pd.DataFrame, pct_stock_1) -> None:
        total_loss_array = get_total_loss_array(df, pct_stock_1)

        self.fitted_model = sorted(total_loss_array)

    def _get_var_without_check(self, alpha: float) -> float:
        n = len(self.fitted_model)

        VaR = self.fitted_model[int(n*alpha)]

        return VaR


class NormalMixtureMethod(VarMethod):

    def _fit_without_check(self, df: pd.DataFrame, pct_stock_1, normal_mixture_model: GaussianMixture = None) -> None:
        total_loss_array = get_total_loss_array(df, pct_stock_1)

        if normal_mixture_model is None:
            normal_mixture_model = GaussianMixture(2)

        self.fitted_model = normal_mixture_model.fit(total_loss_array.reshape(-1,1))

    def _get_var_without_check(self, alpha: float, n_samples: float = 100_000) -> float:
        samples = self.fitted_model.sample(n_samples)[0]
        samples_sorted = sorted(samples)

        VaR = samples_sorted[int(n_samples*alpha)]

        return VaR


class EVTMethod(VarMethod):

    def _fit_without_check(self, df: pd.DataFrame, pct_stock_1: float, k_range = range(1,100), chosen_k = None) -> None:
        loss_array = get_total_loss_array(df, pct_stock_1)

        if chosen_k is None:
            make_hill_plot(loss_array, k_range)
            # make_hill_plot_2(loss_array)
            chosen_k = int(input("What k do you think is suitable?"))

        alpha_hat = hill_estimate(loss_array, chosen_k)
        threshold = sorted(loss_array, reverse=True)[chosen_k]

        self.fitted_model = {"threshold": threshold,
                             "alpha_hat": alpha_hat,
                             "k": chosen_k,
                             "n": len(loss_array)}

    def _get_var_without_check(self, alpha: float) -> float:
        threshold = self.fitted_model["threshold"]
        alpha_hat = self.fitted_model["alpha_hat"]
        k = self.fitted_model["k"]
        n = self.fitted_model["n"]

        VaR = threshold * (k/(n*(1-alpha))) ** (1/alpha_hat)

        return VaR


class GARCHMethod(VarMethod):

    def _fit_without_check(self, df: pd.DataFrame, pct_stock_1: float) -> None:
        loss_array = get_total_loss_array(df, pct_stock_1)

        garch_model = arch.arch_model(loss_array)
        garch_res = garch_model.fit()

        self.fitted_model = garch_res

    def _get_var_without_check(self, alpha: float) -> float:

        volatility = float(self.fitted_model.forecast(horizon=1).variance["h.1"][-1:].iloc[0])

        mu_hat = self.fitted_model.params[0]

        std_resid = self.fitted_model.std_resid

        n = len(std_resid)

        hs_var = sorted(std_resid)[int(n*alpha)]

        VaR = mu_hat + math.sqrt(volatility) * hs_var
        print(f"Predicted volatility: {volatility}")
        print(f"HS VaR for error term: {hs_var}")

        return VaR


class CopulaMethod(VarMethod):
    NSIMS = 100_000

    def _fit_without_check(self, df: pd.DataFrame, pct_stock_1: float) -> None:
        self.pct_stock_1 = pct_stock_1
        columns = df.columns
        x = np.asarray(-1*df[columns[0]], dtype=float)
        y = np.asarray(-1*df[columns[1]], dtype=float)

        # Fit the t distributions
        self.x_params = t.fit(x)
        self.y_params = t.fit(y)

        # Fit theta and tau
        tau_hat = kendalltau(x, y).statistic

        theta_hat = 2/(1-tau_hat) - 2

        # Fit the copula and simulate
        # self.copula = Clayton(theta_hat, n_samples=NSIMS, dim=2)
        self.copula = ClaytonCopula(theta=theta_hat, k_dim=2)

        self.fitted_model = (self.copula, self.x_params, self.y_params, tau_hat, theta_hat)

    def _get_var_without_check(self, alpha: float) -> float:
        simulations = self.copula.rvs(self.NSIMS)

        shape_x, mu_x, scale_x = self.x_params
        shape_y, mu_y, scale_y = self.y_params

        def inv_cdf_x(u):
            return t.ppf(u, shape_x, mu_x, scale_x)

        def inv_cdf_y(u):
            return t.ppf(u, shape_y, mu_y, scale_y)

        vinv_cdf_x = np.vectorize(inv_cdf_x)
        vinv_cdf_y = np.vectorize(inv_cdf_y)

        marginals_x = vinv_cdf_x(simulations[:, 0])
        marginals_y = vinv_cdf_y(simulations[:, 1])

        losses = -1 * (marginals_x * self.pct_stock_1 + marginals_y * (1 - self.pct_stock_1))

        n = len(losses)

        VaR = sorted(losses)[int(n*alpha)]

        return VaR

    def get_upper_tail_dependence(self) -> float:
        check_for_fitted_model(self)

        theta = self.fitted_model[-1]

        copula = archimedean("clayton")

        # Upper tail dependence of the data is the lower tail dependence of the copula
        utdc = copula.LTDC(theta)

        return utdc
