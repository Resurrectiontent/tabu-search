from functools import partial
from numbers import Number
from operator import mul
from typing import Iterable, Callable, Literal

import numpy as np

from tabusearch.solution.quality.aggregated import BaseAggregatedSolutionQualityInfo, \
    CompareAggregatedSolutionQualityInfo, \
    AggregateComparisonSolutionQualityInfo
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.solution.quality.single import SolutionQualityInfo


# TODO: add docstrings

def sum_metrics_aggregation(name: str, weights: list[Number] | None = None) \
        -> Callable[[Iterable[Iterable[SolutionQualityInfo | float]]], list[BaseAggregatedSolutionQualityInfo]]:
    if weights is None:
        aggregation = sum
    else:
        def aggregation(metric_values: Iterable[float]):
            return sum(map(mul, weights, metric_values))

    one_solution_metrics = partial(CompareAggregatedSolutionQualityInfo, name=name, aggregation=aggregation)

    def iter_solutions(solutions_metrics: Iterable[Iterable[SolutionQualityInfo]]) \
            -> list[CompareAggregatedSolutionQualityInfo]:
        return list(map(one_solution_metrics, solutions_metrics))

    return iter_solutions


def normalized_weighted_metrics_aggregation(name: str, weights: list[Number]) \
        -> Callable[[Iterable[Iterable[SolutionQualityInfo]]], list[BaseAggregatedSolutionQualityInfo]]:
    # per-metrics matrix normalization
    def get_weights(solutions_metrics: Iterable[Iterable[SolutionQualityInfo]]) -> list[list[float]]:
        matrix = np.array([[float(metric_value) for metric_value in solution_metrics]
                           for solution_metrics in solutions_metrics])
        normalized = (matrix - matrix.min(axis=0)) / matrix.ptp(axis=0)
        norm_weighted = normalized * weights
        output_weights = (norm_weighted / matrix).tolist()
        return output_weights

    def aggregation(solution_metrics: Iterable[float], output_weights: list[float]) -> float:
        return sum(map(mul, solution_metrics, output_weights))

    one_solution_metrics = partial(CompareAggregatedSolutionQualityInfo, name=name)

    def iter_solutions(solutions_metrics: Iterable[Iterable[SolutionQualityInfo]]) \
            -> list[CompareAggregatedSolutionQualityInfo]:
        output_weights = get_weights(solutions_metrics)
        return [one_solution_metrics(mtrx, aggregation=partial(aggregation, output_weights=wghts))
                for mtrx, wghts in zip(solutions_metrics, output_weights)]

    return iter_solutions


# TODO: re-write from list of metrics on matrix of them
def per_metric_comparison_aggregation(name: str, aggregation: Literal['all', 'any', 'most']
                                                              | Callable[[Iterable[bool]], bool] = 'all') \
        -> Callable[[Iterable[BaseSolutionQualityInfo]], BaseAggregatedSolutionQualityInfo]:
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
