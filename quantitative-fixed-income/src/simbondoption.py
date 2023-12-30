# Goal
'''This python file allows for the simulation of a European
call bond option, for a setting where the underlying bond is
modeled by a 1-factor Vasicek model.'''

# Import packages
import numpy as np  # random numbers
from math import exp, sqrt  # exponential and square root function


# Functions
def get_vasicek_price(kappa: float,
                      mu: float,
                      sigma: float,
                      r: float,
                      tau: float) -> float:
    '''
    get_vasicek_price obtains the Vasicek price of a bond under
    risk-neutrality i.e. (lambda = 0).

    param kappa: short-rate speed of mean reversion
    param mu: long-run mean short-rate
    param sigma: short-rate volatility
    param r: short-rate
    param tau: time to maturity of bond
    return: Vasicek price of bond'''

    # Calculate B(tau) and A(tau)
    B = (exp(-kappa*tau) - 1) / kappa

    A = (B + tau) * ((sigma ** 2) / (2 * (kappa ** 2)) - mu) \
        - (sigma ** 2) * (B ** 2) / (4 * kappa)

    # Return the vasicek price
    return exp(A + B*r)


def main(kappa: float,
         mu: float,
         sigma: float,
         r0: float,
         R: int,
         Delta: float,
         T0: int,
         T1: int,
         K: float) -> float:
    '''
    main obtains the European call bond option price using simulation, where
    the price of the bond is determined by a 1-factor Vasicek model.

    param kappa: short-rate speed of mean reversion
    param mu: long-run mean short-rate
    param sigma: short-rate volatility
    param r0: initial short-rate
    param R: number of replications in simulation study
    param Delta: Euler step-length
    param T0: expiry of bond option
    param T1: maturity of bond
    param K: strike price
    return: simulated price of European call bond option
    '''

    # Get number of steps used for the short rate path
    n_steps = int(T0 / Delta)

    # Initialize lists of simulated prices
    simulated_prices = list()

    # Perform the simulation R times
    for _ in range(R):

        # Initialize the list of simulated short rates
        r = [r0]

        # Obtain the simulated short rate path
        for _ in range(1, n_steps):

            # Draw a random normal with the correct variance
            e = np.random.normal(0, 1)
            z = e * sqrt((sigma**2) * (1-exp(-2*kappa*Delta)) / (2*kappa))

            # Add the next simulated short rate to the list
            r_t = exp(-kappa*Delta) * r[-1] + (1 - exp(-kappa*Delta)) * mu + z
            r.append(r_t)

        # Calculate the bond price given the simulated short rate path
        r_T0 = r[-1]
        vasicek_price = get_vasicek_price(kappa, mu, sigma, r_T0, T1 - T0)

        # Calculate the discounted pay-off (=price) and append to the list
        discounted_pay_off = exp(-sum(r) * Delta) * max(vasicek_price - K, 0)
        simulated_prices.append(discounted_pay_off)

    # Return the mean of the simulated prices
    return sum(simulated_prices) / len(simulated_prices)
