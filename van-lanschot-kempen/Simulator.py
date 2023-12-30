import math
import numpy as np


class Simulator:
    """
    A class that simulates the performance of a financial instrument.

    Attributes:
        note (Note): The financial instrument being simulated.
        r (float): The risk-free interest rate.
        days (int): The number of simulation days.
        underlying_return_drawer (function): A function that generates
                                            underlying return values.

    Methods:
        get_simulated_returns(): Generates a list of simulated underlying
                                return values.
        get_simulated_underlying_values(): Generates a list of simulated
                                            underlying values.
        simulate_single_discounted_payoff(): Simulates a single discounted
                                            payoff.
        simulate_multiple_discounted_payoffs(n, seed=None): Simulates multiple
                                                            discounted payoffs.
        get_simulated_price(n, seed=None): Calculates the simulated price
                                            of the financial instrument.
    """

    daysperyear = 250

    def __init__(self, note, r, underlying_return_drawer):
        """
        Initializes a Simulator object.

        Args:
            note (Note): The financial instrument being simulated.
            r (float): The risk-free interest rate.
            days (int): The number of simulation days.
            underlying_return_drawer (function): A function that generates
                                                underlying return values.
        """
        self.note = note
        self.r = r
        self.underlying_return_drawer = underlying_return_drawer
        self.days = self.note.maturity * self.daysperyear

    def get_simulated_returns(self) -> list[float]:
        """
        Generates a list of simulated underlying return values.

        Returns:
            list[float]: The simulated underlying return values.
        """
        return [self.underlying_return_drawer() for _ in range(self.days)]

    def get_simulated_underlying_values(self) -> list[float]:
        """
        Generates a list of simulated underlying values.

        Returns:
            list[float]: The simulated underlying values.
        """
        underlying_values = [self.note.nominal_value]

        for r in self.get_simulated_returns():
            underlying_values.append(underlying_values[-1] * r)

        return underlying_values

    def simulate_single_discounted_payoff(self) -> float:
        """
        Simulates a single discounted payoff.

        Returns:
            float: The discounted payoff value.
        """
        underlying_values = self.get_simulated_underlying_values()
        discounted_payoff = 0

        for d, uv in enumerate(underlying_values):
            # Set new underlying value
            self.note.underlying_value = uv

            # Coupon moment
            if self.note.coupon and d in self.note.coupon_days:
                coupon = self.note.coupon_moment()
                discount_factor = math.exp(-self.r * d / self.daysperyear)
                discounted_payoff += discount_factor * coupon

        # End settlement
        end_settlement = self.note.end_settlement
        discount_factor = math.exp(-self.r * d / self.daysperyear)
        discounted_payoff += discount_factor * end_settlement

        return discounted_payoff

    def simulate_multiple_discounted_payoffs(self, n: int, seed: int = None):
        """
        Simulates multiple discounted payoffs.

        Args:
            n (int): The number of payoffs to simulate.
            seed (int, optional): The seed for the random number generator.
                Defaults to None.

        Returns:
            list[float]: The list of simulated discounted payoffs.
        """
        if seed is not None:
            np.random.seed(seed)

        return [self.simulate_single_discounted_payoff() for _ in range(n)]

    def get_simulated_price(self, n: int, seed: int = None):
        """
        Calculates the simulated price of the financial instrument.

        Args:
            n (int): The number of simulations to perform.
            seed (int, optional): The seed for the random number generator.
                Defaults to None.

        Returns:
            float: The simulated price of the financial instrument.
        """
        discounted_payoffs = self.simulate_multiple_discounted_payoffs(n, seed)

        return sum(discounted_payoffs) / len(discounted_payoffs)
