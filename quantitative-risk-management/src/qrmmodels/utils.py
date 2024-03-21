from __future__ import annotations
import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt

from .exceptions import WrongDataFrameError, UnfittedModelError

from functools import lru_cache
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .models.var_method import VarMethod


def load_data_from_path(path: str, sheet_name: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path)

    sheet = wb[sheet_name]

    df = pd.DataFrame(sheet.values)
    df = df[df.columns[:2]]
    df.columns = df.iloc[0]
    df = df[1:]

    return df


def get_total_loss_array(df: pd.DataFrame, pct_stock_1: float):

    columns = df.columns

    total_losses = pct_stock_1 * df[columns[0]] + (1-pct_stock_1) * df[columns[1]]

    return total_losses.to_numpy().astype('float')


def check_for_bad_df(df: pd.DataFrame):

    if len(df.columns) != 2:
        raise WrongDataFrameError(f"The DataFrame shoudl have 2 columns, it has {len(df.columns)}")


def check_for_fitted_model(method: VarMethod):

    if method.fitted_model is None:
        raise UnfittedModelError("You should first fit the model")


def hill_estimate(losses: np.array, k: int):

    X = peaks_including_threshold(losses, k)

    sum = 0
    for i in range(0, k):
        sum += (np.log(X[i]) - np.log(X[-1]))

    alpha_inv = sum / k

    return 1 / alpha_inv


def peaks_including_threshold(losses: np.array, k: int):
    losses_sorted = sorted(losses, reverse=True)
    return losses_sorted[:(k+1)]


def make_hill_plot(losses: np.array, k_range):
    alpha_hats = []
    for k in k_range:
        alpha_hats.append(hill_estimate(losses, k))

    plt.plot(k_range, alpha_hats)
    plt.show()


def make_tail_dependence_plot(data: pd.DataFrame, k_range):
    check_for_bad_df(data)

    data_array = data.to_numpy()
    x1 = data_array[:, 0]
    x2 = data_array[:, 1]

    ordered_x1 = sorted(x1)
    ordered_x2 = sorted(x2)

    lambda_hats = []
    for k in k_range:
        lambda_hats.append(lambda_hat(ordered_x1, ordered_x2, k))

    plt.plot(k_range, lambda_hats)
    plt.show()


def lambda_hat(ordered_x1: np.array, ordered_x2: np.array, k: int) -> float:
    sum = 0
    n = len(ordered_x1)

    for t in range(n):
        if (ordered_x1[t] > ordered_x2[n-k]) and (ordered_x2[t] > ordered_x1[n-k]):
            sum += 1

    return sum / k
