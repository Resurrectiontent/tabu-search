from functools import partial
from numbers import Number
from typing import Union, Iterable, Callable

from numpy import ndarray
from numpy.typing import NDArray

from tabusearch.solution.quality.aggregated import BaseAggregatedSolutionQualityInfo, \
    CompareAggregatedSolutionQualityInfo, \
    AggregateComparisonSolutionQualityInfo
from tabusearch.solution.quality.base import BaseSolutionQualityInfo


# TODO: add docstring


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


def per_metric_comparison_aggregation(name: str, aggregation: Union[str, Callable[[Iterable[bool]], bool]] = 'all'):
    def get_aggregation(s: str):
        if s == 'all':
            return all
        if s == 'any':
            return any
        if s == 'most':
            return lambda l: len([None for x in l if x]) / len(l)
        raise ValueError(f'Argument aggregation should be "all", "any", "most" or callable. Was "{s}".')

    try:
        aggregation = get_aggregation(aggregation) if isinstance(aggregation, str) else aggregation
    except ValueError as ve:
        raise ve

    return partial(AggregateComparisonSolutionQualityInfo, name=name, aggregation=aggregation)
