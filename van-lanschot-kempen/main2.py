from ParticipatieNote import ParticipatieNote
from Simulator import Simulator
from scipy.stats import lognorm  # lognormal distribution


def drawer(inner_func, *args, **kwargs):
    def outer_func():
        return inner_func(*args, **kwargs)

    return outer_func


def main():
    note = ParticipatieNote(1, 1000, 900, True, 1.2)

    return_drawer = drawer(lognorm.rvs, 0.1)
    simulator = Simulator(note, 0.02, return_drawer)
    print(simulator.get_simulated_price(n=10))


if __name__ == "__main__":
    main()
