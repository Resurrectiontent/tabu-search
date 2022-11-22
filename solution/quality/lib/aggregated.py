from functools import partial
from numbers import Number
from typing import Union, Iterable, Callable

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from solution.quality.aggregated import BaseAggregatedSolutionQualityInfo, CompareAggregatedSolutionQualityInfo
from solution.quality.base import BaseSolutionQualityInfo


def sum_metrics_aggregation(name: str, weights: Union[NDArray[Number], Iterable[Number], None] = None) \
        -> Callable[[Iterable[BaseSolutionQualityInfo]], BaseAggregatedSolutionQualityInfo]:
    if weights is None:
        aggregation = sum

    elif isinstance(weights, ndarray):
        def aggregation(metric_values: Union[NDArray[float], Iterable[float]]):
            if isinstance(metric_values, ndarray):
                return metric_values * weights
            return sum([i * j for i, j in zip(metric_values, list(weights))])

    else:
        def aggregation(metric_values: Union[NDArray[float], Iterable[float]]):
            metric_values = list(metric_values) if isinstance(metric_values, ndarray) else metric_values
            return sum([i * j for i, j in zip(metric_values, list(weights))])

    return partial(CompareAggregatedSolutionQualityInfo, name=name, aggregation=aggregation)
