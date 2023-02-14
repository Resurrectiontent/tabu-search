import numpy as np
from scipy.stats import norm, expon
from matplotlib import pyplot as plt

from solution.base import Solution
from tabusearch import TabuSearch


def rosenbrock(x: np.ndarray) -> float:
    """
    The Rosenbrock function of n dimensions.
    Args:
        x (numpy.ndarray): The vector of n independent variables
    Returns:
        float: The value of the Rosenbrock function at x
    """
    n = x.size
    if n < 2:
        raise ValueError("x must be an array of at least size 2")
    f = 0
    for i in range(1, n):
        f += 100 * (x[i] - x[i-1]**2)**2 + (1 - x[i-1])**2
    return f


def test_rosenbrock_optimisation():
    x0 = norm.rvs(loc=1, scale=3.5, size=12).astype(int)  # np.array([-1, 0, 1, 2, 3])
    optimiser = TabuSearch(quality=rosenbrock,
                           tabu_time=4,
                           selection=lambda _: expon.rvs(size=1).astype(int)[0])
    s = optimiser.optimize(x0)
    h: list[Solution] = optimiser._history
    print('\n', s.position)
    print(s.quality)
    plt.plot(np.linspace(0, 1, len(h)), np.array([float(x.quality) for x in h]))
    plt.show()
