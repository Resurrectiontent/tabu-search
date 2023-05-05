from collections import deque
from typing import Iterable, Callable

import numpy as np
from numpy.typing import NDArray

from tabusearch.typing_ import TData
from tabusearch.memory.evaluating.base import SecondOrderEvaluatingMemoryCriterion
from tabusearch.solution.base import Solution
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.solution.quality.single import SolutionQualityInfo


# TODO: implement option to calculate gradient for 3 historical points and compare it with direction of the new point
class GradientAccelerator(SecondOrderEvaluatingMemoryCriterion[TData, NDArray]):
    _history_points: deque[NDArray]
    _history_values: deque[float]
    _allow_2p_grad: bool

    def __init__(self, data_converter: Callable[[Iterable[TData]], Iterable[NDArray]] | None = None,
                 allow_2point_gradient: bool = False):
        self._history_points = deque(maxlen=2)
        self._history_values = deque(maxlen=2)
        self._allow_2p_grad = allow_2point_gradient

        if data_converter:
            self.convert_to_evaluated_data_type = data_converter

    def memorize(self, move: Solution):
        [position] = self.convert_to_evaluated_data_type([move.position])

        if not isinstance(position, np.ndrray):
            raise Exception(f'Converted data was not ndarray. It was {type(position)}')
        if not isinstance(move.quality, SolutionQualityInfo):
            raise Exception('Quality of solution should be SolutionQualityInfo descendant'
                            ' to use in GradientAccelerator.')

        self._history_points.append(position)
        self._history_values.append(move.quality.value)

    def evaluate(self, x: Iterable[tuple[NDArray, SolutionQualityInfo]]) -> Iterable[BaseSolutionQualityInfo]:
        match self._allow_2p_grad, len(self._history_points):
            case True, 1:
                weights = [sum(pseudogradient_2p(self._history_points[0], d,
                                                          self._history_values[0], q)) for d, q in x]
            case _, 2:
                weights = [sum(pseudogradient(list(self._history_points) + [d],
                                                       list(self._history_values) + [q])) for d, q in x]
            case _:
                weights = [1 for _ in x]

        return [q.quality_like(data=d, name='GradientAccelerator', float_=w, value_str=str)
                for (d, q), w in zip(x, weights)]


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
