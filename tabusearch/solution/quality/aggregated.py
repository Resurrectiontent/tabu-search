from abc import ABC
from functools import partialmethod
from operator import eq, lt
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.solution.quality.single import SolutionQualityInfo


# TODO: consider (conditional) making *metrics in ctor args lazy and BaseSolutionQualityInfo

class BaseAggregatedSolutionQualityInfo(BaseSolutionQualityInfo, ABC):
    """
    Base class for SolutionQualityInfo aggregation.
    Implements no abstract logic. Created for marking SolutionQualityInfo aggregating classes via inheritance.
    Inherited classes should also
    """
    pass


class AggregateComparisonSolutionQualityInfo(BaseAggregatedSolutionQualityInfo):
    def __init__(self, *metrics, name: str, aggregation: [Callable[[Iterable[bool]], bool]]):
        value_str = '\n'.join(list(map(str, metrics)))
        super(BaseAggregatedSolutionQualityInfo, self).__init__(name, value_str)

        self._data = list(metrics)
        self._aggregation = aggregation

    @property
    def _float_ndarray(self) -> NDArray[float]:
        return np.array([float(m) for m in self._data])

    def _cmp_agg(self, other: 'AggregateComparisonSolutionQualityInfo', cmp):
        return self._aggregation(cmp(self._float_ndarray, other._float_ndarray))

    _equals_to = partialmethod(_cmp_agg, cmp=eq)
    _less_than = partialmethod(_cmp_agg, cmp=lt)


class CompareAggregatedSolutionQualityInfo(SolutionQualityInfo, BaseAggregatedSolutionQualityInfo):
    def __init__(self, *metrics, name: str, aggregation: Callable[[Iterable[float]], float]):
        # may remain unused
        def value_str(metr):
            return '\n'.join(list(map(str, metr)))
        # most likely will be used
        float_v = aggregation(map(float, metrics))
        super().__init__(metrics, name, float_v, False, value_str)
