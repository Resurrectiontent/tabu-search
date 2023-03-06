from numbers import Number

import numpy as np
from numpy.typing import NDArray

from tests.lib.functions.constrains import shape, bounds


def styblinski_tang(x: NDArray[Number]):
    return (x**4 - 16*x**2 + 5*x).sum() / 2


def rosenbrock(x: NDArray[Number]):
    return (100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0).sum()


@shape((2,))
@bounds(np.array([-1.5, -3]), np.array([4, 4]))
def mccormick(x: NDArray[Number]):
    return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1


def michalewicz(x):
    return -np.sum(np.sin(x)*(np.sin((x**2)/np.pi))**20)


@bounds(np.array([-5]), np.array([10]))
def zakharov(x):
    return np.sum(x**2) + np.sum(0.5*(1+np.arange(len(x)))*x)**2 + np.sum(0.5*(1+np.arange(len(x)))*x)**4
