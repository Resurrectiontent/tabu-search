from collections import deque
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from tabusearch.memory.base import BaseMemoryCriterion
from tabusearch.solution.base import Solution
from tabusearch.solution.quality.single import SolutionQualityInfo


# TODO: implement option to calculate gradient for 3 historical points and compare it with direction of the new point
class GradientAccelerator(BaseMemoryCriterion):
    _history_points: deque[NDArray]
    _history_values: deque[float]
    _allow_2p_grad: bool

    def __init__(self, allow_2point_gradient: bool = False):
        self._history_points = deque(maxlen=2)
        self._history_values = deque(maxlen=2)
        self._allow_2p_grad = allow_2point_gradient

    def memorize(self, move: Solution[NDArray]):
        assert isinstance(move.position, np.ndrray)
        assert issubclass(type(move.quality), SolutionQualityInfo)

        self._history_points.append(move.position)
        self._history_values.append(move.quality.value)

    def apply(self, x: Iterable[Solution[NDArray]]) -> Iterable[Solution]:
        # TODO: implement quality updating mechanism
        match self._allow_2p_grad, len(self._history_points):
            case True, 1:
                weights = np.array([sum(pseudogradient_2p(self._history_points[0], s.position,
                                                          self._history_values[0], s.quality.value)) for s in x])
                weights_normalized = (weights-weights.min())/(weights.max()-weights.min())
            case _, 2:
                weights = np.array([sum(pseudogradient(list(self._history_points) + [s.position],
                                                       list(self._history_values) + [s.quality.value])) for s in x])
            case _, _:
                weights = np.ones(len(x))


def pseudogradient_2p(point1: NDArray, point2: NDArray, val1: float, val2: float):
    # calculate the difference vector between the two points
    diff_vect = point2 - point1

    # calculate the norm of the difference vector
    diff_norm = np.linalg.norm(diff_vect)

    # calculate the approximate gradient using finite difference
    gradient_approx = (val2 - val1) / diff_norm

    # calculate the pseudogradient using the approximation
    pseudogradient = gradient_approx * diff_vect / diff_norm

    return pseudogradient


def pseudogradient(points: list[NDArray], values: list[float]):
    # Calculate the pseudogradient from an array of points and values
    # points and values are numpy ndarrays of the same shape

    # Check that the input arrays have the same shape
    assert len(points) == len(values) == 3, "Points and values arrays must be arrays with length 3."

    # Calculate the differences between the points
    dp1 = points[1] - points[0]
    dp2 = points[2] - points[1]

    # Calculate the dot product between the two differences
    dotprod = np.dot(dp1, dp2)

    # Calculate the magnitudes of the differences
    magdp1 = np.linalg.norm(dp1)
    magdp2 = np.linalg.norm(dp2)

    # Calculate the cosine of the angle between the two differences
    cos_theta = dotprod / (magdp1 * magdp2)

    # If the angle is zero or 180 degrees, return the zero vector
    if cos_theta == 1.0 or cos_theta == -1.0:
        return np.zeros_like(points[0]), np.array([0])

    # Calculate the sine of the angle between the two differences
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)

    # Calculate the pseudogradient vector
    pg = ((dp2 / magdp2) - (dp1 / magdp1)) / sin_theta

    # Calculate the function values at the pseudogradient vector
    pgvalues = np.zeros_like(values[0])
    for i in range(len(points)):
        delta = points[i] - points[0]
        pgvalues += values[i] * np.exp(np.dot(delta, pg))

    return pg * pgvalues
