import numpy as np
import pytest
from scipy.stats import norm, expon
from matplotlib import pyplot as plt

from solution.base import Solution
from tabusearch import TabuSearch
from tests.lib.functions import rosenbrock, styblinski_tang, mccormick, michalewicz, zakharov

DEFAULT_TEST_SIZE = 10
DEFAULT_STD = 5


@pytest.mark.parametrize('function', [rosenbrock,
                                      styblinski_tang,
                                      mccormick,
                                      michalewicz,
                                      zakharov])
def test_optimisation(function):
    shape = function.shape if hasattr(function, 'constrained_shape') else (DEFAULT_TEST_SIZE, )

    if hasattr(function, 'constrained_values'):
        min_bounds, max_bounds = (function.min_bounds, function.max_bounds) \
            if function.min_bounds.shape == shape \
            else (np.ones(shape) * function.min_bounds[0],
                  np.ones(shape) * function.max_bounds[0])

        x0 = np.random.random(shape) * (max_bounds - min_bounds) + min_bounds
    else:
        x0 = norm.rvs(scale=DEFAULT_STD, size=shape)

    x0 = x0.astype(int)

    optimiser = TabuSearch(quality=function,
                           max_iter=1000,
                           tabu_time=np.prod(shape),
                           selection=lambda collection_len: min(expon.rvs(size=1).astype(int)[0], collection_len - 1))
    s = optimiser.optimize(x0)
    h: list[Solution] = optimiser._history
    print('\n', s.position)
    print(s.quality)
    plt.plot(np.linspace(0, 1, len(h)), np.array([float(x.quality) for x in h]))
    plt.show()
