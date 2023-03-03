import numpy as np
from scipy.stats import norm, expon
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from solution.base import Solution
from tabusearch import TabuSearch
from tests.lib.functions import rosenbrock, styblinski_tang, mccormick, michalewicz

DEFAULT_TEST_SIZE = 10


def test_optimisation(function):
    if hasattr(function, 'constrained_values'):
        if hasattr(function, 'constrained_shape'):
            size = np.prod(function.shape)


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
    x0 = norm.rvs(loc=0, scale=3, size=50).astype(int)  # np.array([-1, 0, 1, 2, 3])
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


def test_mccormick_optimisation():
    x0 = norm.rvs(loc=1, scale=2, size=2).astype(int)  # np.array([-1, 0, 1, 2, 3])
    optimiser = TabuSearch(quality=mccormick,
                           max_iter=1000,
                           tabu_time=2,
                           selection=lambda collection_len: min(expon.rvs(size=1).astype(int)[0], collection_len - 1))
    s = optimiser.optimize(x0)
    h: list[Solution] = optimiser._history
    print('\n', s.position)
    print(s.quality)
    plt.plot(np.linspace(0, 1, len(h)), np.array([float(x.quality) for x in h]))
    plt.show()


def test_michalewicz_optimisation():
    x0 = norm.rvs(loc=1, scale=2, size=50).astype(int)  # np.array([-1, 0, 1, 2, 3])
    optimiser = TabuSearch(quality=michalewicz,
                           max_iter=1000,
                           tabu_time=2,
                           selection=lambda collection_len: min(expon.rvs(size=1).astype(int)[0], collection_len - 1))
    s = optimiser.optimize(x0)
    h: list[Solution] = optimiser._history
    print('\n', s.position)
    print(s.quality)
    plt.plot(np.linspace(0, 1, len(h)), np.array([float(x.quality) for x in h]))
    plt.show()


def test_scipy_styblinsky_tang():
    vector_length = 50

    x0 = norm.rvs(loc=0, scale=3, size=vector_length).astype(int)
    r = minimize(fun=styblinski_tang, x0=x0, bounds=[(-5, 5)] * vector_length)
    print(f'Finished successfully: {r.success}'
          f'\nGotten result:\n\t{r.x}'
          f'\nFun val: {styblinski_tang(np.array(r.x))}')
