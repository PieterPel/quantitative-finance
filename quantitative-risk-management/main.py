"""
Main script of our assignment.
The code of our qrmmodels package can be found under /src
For installation see the readme.md
"""

# Import packages
import qrmmodels
from qrmmodels.models.var_method import VarMethod
from scipy.stats import jarque_bera
import pandas as pd
import os
import pycop

# Set up constants
VALUE_IN_STOCK_1 = 30 * 10**6
VALUE_IN_STOCK_2 = 70 * 10**6
PCT_STOCK_1 = VALUE_IN_STOCK_1 / (VALUE_IN_STOCK_2 + VALUE_IN_STOCK_1)
ALPHA = 0.99


def main() -> None:
    data_path = get_data_path()
    sheet_name = "Data"

    stock_df = qrmmodels.load_data_from_path(data_path, sheet_name)

    print_method_1(stock_df, PCT_STOCK_1, ALPHA)
    print_method_2(stock_df, PCT_STOCK_1, ALPHA)
    print_method_3(stock_df, PCT_STOCK_1, ALPHA)
    print_method_4(stock_df, PCT_STOCK_1, ALPHA)
    print_method_5(stock_df, PCT_STOCK_1, ALPHA)
    print_method_6(stock_df, PCT_STOCK_1, ALPHA)


47


def get_data_path():
    notebook_dir = os.getcwd()

    data_file_path = os.path.join(notebook_dir, "525573_497053.xlsx")

    return data_file_path


def print_var_of_method(
    method: VarMethod, data: pd.DataFrame, pct_stock_1: float, alpha: float
) -> None:

    method.fit(data, pct_stock_1)

    VaR = method.get_var(alpha)

    print(f"VaR: {VaR}\n")


def print_method_1(
    data: pd.DataFrame, pct_stock_1: float, alpha: float
) -> None:
    from qrmmodels.models.var_method import VarCovarMethod

    print(f"{'Method 1' :-<100}")
    method = VarCovarMethod()
    print_var_of_method(method, data, pct_stock_1, alpha)

    print(method.fitted_model)


def print_method_2(
    data: pd.DataFrame, pct_stock_1: float, alpha: float
) -> None:
    from qrmmodels.models.var_method import HistoricalSimulationMethod

    print(f"{'Method 2' :-<100}")
    method = HistoricalSimulationMethod()
    print_var_of_method(method, data, pct_stock_1, alpha)


def print_method_3(
    data: pd.DataFrame, pct_stock_1: float, alpha: float
) -> None:
    from qrmmodels.models.var_method import NormalMixtureMethod

    print(f"{'Method 3' :-<100}")

    return_array = qrmmodels.utils.get_total_loss_array(data, pct_stock_1)
    print(f"Jarque-Bera: {jarque_bera(return_array)}")

    method = NormalMixtureMethod()
    print_var_of_method(method, data, pct_stock_1, alpha)

    print(f"Means: \n {method.fitted_model.means_} \n")
    print(f"Covariances: \n {method.fitted_model.covariances_} \n")
    print(f"weights: \n {method.fitted_model.weights_} \n")


def print_method_4(
    data: pd.DataFrame, pct_stock_1: float, alpha: float
) -> None:
    from qrmmodels.models.var_method import EVTMethod

    print(f"{'Method 4' :-<100}")
    method = EVTMethod()
    print_var_of_method(method, data, pct_stock_1, alpha)

    print(method.fitted_model)


def print_method_5(
    data: pd.DataFrame, pct_stock_1: float, alpha: float
) -> None:
    from qrmmodels.models.var_method import GARCHMethod

    print(f"{'Method 5' :-<100}")
    method = GARCHMethod()
    print_var_of_method(method, data, pct_stock_1, alpha)

    for name, param in zip(
        method.fitted_model._names, method.fitted_model.params
    ):
        print(f"{name}: {param}")


def print_method_6(
    data: pd.DataFrame, pct_stock_1: float, alpha: float
) -> None:
    from qrmmodels.models.var_method import CopulaMethod

    print(f"{'Method 6' :-<100}")
    method = CopulaMethod()
    print_var_of_method(method, data, pct_stock_1, alpha)

    # Print parameters of t distributions
    names = ["df", "mean", "sigma"]
    print("Stock 1:")
    for name, param in zip(names, method.x_params):
        print(f"{name}: {param}")

    print("\nStock 2")
    for name, param in zip(names, method.y_params):
        print(f"{name}: {param}")

    print(f"tau-hat: {method.fitted_model[-2]}")
    print(f"theta-hat: {method.fitted_model[-1]}")

    print(f"Upper tail dependence: {method.get_upper_tail_dependence()}")

    from qrmmodels.utils import make_tail_dependence_plot

    k_range = range(1, 1000)
    make_tail_dependence_plot(data, k_range)

    cop = pycop.empirical(data)
    print(cop.optimal_tdc("upper"))


if __name__ == "__main__":
    main()
