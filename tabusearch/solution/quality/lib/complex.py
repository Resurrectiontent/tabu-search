from typing import Callable

from tabusearch.solution.quality.aggregated import BaseAggregatedSolutionQualityInfo
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.solution.quality.complex import ComplexSolutionQualityInfo
from tabusearch.solution.quality.single import SolutionQualityInfo
from tabusearch.typing_ import TData


def complex_metric(*metrics: Callable[[list[TData]], list[SolutionQualityInfo]],
                   aggregation: Callable[[list[list[BaseSolutionQualityInfo]]],
                                         list[BaseAggregatedSolutionQualityInfo]]):
    """

    :param metrics: Accounted metrics. The first one is considered as the main one
    :param aggregation: Metrics aggregation.
      Should process the whole "matrix" of metrics - list with list of metrics (for each item in data)
    :return:
    """
    def generate_complex_metric(x: list[TData]) -> list[ComplexSolutionQualityInfo]:
        # TODO: consider removing `list(i)`, because it just satisfies type hints
        single_metrics: list[list[BaseSolutionQualityInfo]] = [list(i) for i in zip(*[metric(x) for metric in metrics])]
        aggregated_metric = aggregation(single_metrics)
        return [ComplexSolutionQualityInfo(main, full) for [main, *_], full in zip(single_metrics, aggregated_metric)]

    return generate_complex_metric
