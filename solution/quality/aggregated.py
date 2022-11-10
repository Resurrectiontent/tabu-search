from functools import partialmethod
from operator import eq, lt
from typing import Callable, Iterable

import numpy as np
from numpy import ndarray

from solution.quality.base import BaseSolutionQualityMetric
from solution.quality.single import SolutionQualityMetric

# TODO: consider introducing convenient shortcuts for popular aggregations and comparisons


class AggregateComparisonSolutionQuality(BaseSolutionQualityMetric):
    def __init__(self, name: str, *metrics, aggregation: [Callable[[Iterable[bool]], bool]]):
        value_str = '\n'.join(list(map(str, metrics)))
        super().__init__(name, value_str)

        self._data = list(metrics)
        self._aggregation = aggregation

    @property
    def _float_ndarray(self) -> ndarray:
        return np.array([float(m) for m in self._data])

    def _cmp_agg(self, other: 'AggregateComparisonSolutionQuality', cmp):
        return self._aggregation(cmp(self._float_ndarray, other._float_ndarray))

    _equals_to = partialmethod(_cmp_agg, cmp=eq)
    _less_than = partialmethod(_cmp_agg, cmp=lt)


class CompareAggregatedSolutionQuality(SolutionQualityMetric):
    def __init__(self, name: str, *metrics, aggregation: Callable[[Iterable[float]], float]):
        # may remain unused
        def value_str(metr):
            return '\n'.join(list(map(str, metr)))
        # most likely will be used
        float_v = aggregation(map(float, metrics))
        super().__init__(name, metrics, float_v, False, value_str)
