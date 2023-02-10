import numpy as np

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
    x0 = np.array([0.5, 1, 1.25, 0.25, 0.01])
    optimiser = TabuSearch(quality=rosenbrock)
    s = optimiser.optimize(x0)
    print(s.__dict__)
