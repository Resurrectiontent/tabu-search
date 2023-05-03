from typing import Callable, Iterable, Generic

from tabusearch.solution.quality.aggregated import BaseAggregatedSolutionQualityInfo
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.typing_ import TData


class SolutionQualityFactory(Generic[TData]):
    _is_aggregated: bool

    _metrics: list[Callable[[list[TData]], list[BaseSolutionQualityInfo]]]
    _metrics_aggregation: Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]],
                                   Iterable[BaseAggregatedSolutionQualityInfo]] | None

    # TODO: consider moving ctor logics to helper functions
    #  and taking final `(ndarray) -> BaseSolutionQualityInfo` callable as an __init__ argument
    def __init__(self, *metrics: Callable[[list[TData]], list[BaseSolutionQualityInfo]],
                 metrics_aggregation: Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]],
                                               Iterable[BaseAggregatedSolutionQualityInfo]]
                                      | None = None):
        match metrics, metrics_aggregation:
            case [_], None:
                self._is_aggregated = False
            case [_, _, *_], _:
                self._is_aggregated = True
            case _, _:
                raise Exception('SolutionQualityFactory should be initialised either with one metric '
                                'or with several metrics and an aggregation.')

        self._metrics = list(metrics)
        self._metrics_aggregation = metrics_aggregation

    def __call__(self, x: list[TData]) -> Iterable[BaseSolutionQualityInfo]:
        return self._apply_aggregation(x) if self._is_aggregated else self._apply_single_metric(x)

    def single(self, x: TData) -> BaseSolutionQualityInfo:
        """
        Generates quality metric for single data instance
        :param x: Single data instance
        :return: Solution quality for data instance
        """
        [result] = self.__call__([x])
        return result

    def add_metrics(self, *metrics: Callable[[list[TData]], list[BaseSolutionQualityInfo]]):
        if not self._is_aggregated:
            raise Exception('Cannot add metrics to not aggregating factory.')

        self._metrics.extend(metrics)

    def aggregate(self, metrics_aggregation: Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]],
                                                      Iterable[BaseAggregatedSolutionQualityInfo]],
                  *metrics: Callable[[list[TData]], list[BaseSolutionQualityInfo]]):
        if self._is_aggregated:
            raise Exception('Cannot add aggregation to an already aggregated factory.')
        if not metrics:
            raise Exception('Should add at least one metric for aggregation.')
        self._is_aggregated = True
        self._metrics_aggregation = metrics_aggregation
        self._metrics.extend(metrics)

    # TODO: think-off params and implement logics
    def wrap_aggregate(self):
        ...

    def _apply_aggregation(self, x: list[TData]) -> Iterable[BaseSolutionQualityInfo]:
        return self._metrics_aggregation(zip(*[m(x) for m in self._metrics]))

    def _apply_single_metric(self, x: list[TData]) -> Iterable[BaseSolutionQualityInfo]:
        [metric] = self._metrics
        return metric(x)
