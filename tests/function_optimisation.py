import numpy as np
from scipy.stats import norm, expon
from matplotlib import pyplot as plt

from solution.base import Solution
from tabusearch import TabuSearch
from tests.lib.functions import rosenbrock, styblinski_tang


def test_rosenbrock_optimisation():
    x0 = norm.rvs(loc=1, scale=3.5, size=50).astype(int)  # np.array([-1, 0, 1, 2, 3])
    optimiser = TabuSearch(quality=rosenbrock,
                           max_iter=1000,
                           tabu_time=35,
                           selection=lambda _: expon.rvs(size=1).astype(int)[0])
    s = optimiser.optimize(x0)
    h: list[Solution] = optimiser._history
    print('\n', s.position)
    print(s.quality)
    plt.plot(np.linspace(0, 1, len(h)), np.array([float(x.quality) for x in h]))
    plt.show()


# TODO: pass func name in ctor
def test_styblinski_tang_optimisation():
    x0 = norm.rvs(loc=0, scale=3, size=6).astype(int)  # np.array([-1, 0, 1, 2, 3])
    optimiser = TabuSearch(quality=styblinski_tang,
                           max_iter=1000,
                           tabu_time=3,
                           selection=lambda _: expon.rvs(size=1).astype(int)[0])
    s = optimiser.optimize(x0)
    h: list[Solution] = optimiser._history
    print('\n', s.position)
    print(s.quality)
    plt.plot(np.linspace(0, 1, len(h)), np.array([float(x.quality) for x in h]))
    plt.show()
