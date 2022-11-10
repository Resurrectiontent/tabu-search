from functools import partialmethod
from operator import eq, lt
from typing import Callable, Iterable

import numpy as np
from numpy import ndarray

from solution.quality.base import BaseSolutionQualityMetric


class AggregatedSolutionQuality(BaseSolutionQualityMetric):
    def __init__(self, name: str, *metrics, aggregation: [Callable[[Iterable[bool]], bool]]):
        value_str = '\n'.join(list(map(str, metrics)))
        super().__init__(name, value_str)

        self._metrics = list(metrics)
        self._aggregation = aggregation

    @property
    def _float_ndarray(self) -> ndarray:
        return np.array([float(m) for m in self._metrics])

    def _cmp_agg(self, other: 'AggregatedSolutionQuality', cmp):
        return self._aggregation(cmp(self._float_ndarray, other._float_ndarray))

    _equals_to = partialmethod(_cmp_agg, cmp=eq)
    _less_than = partialmethod(_cmp_agg, cmp=lt)

# TODO: Custom aggregation with aggregation:[Callable[[Iterable[bool]], bool]
# TODO: Shortcut for all, any, etc.
