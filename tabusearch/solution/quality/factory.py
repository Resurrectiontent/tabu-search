from functools import partial
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
        Generates quality metric for single data instance.
        :param x: Single data instance.
        :return: Solution quality for data instance.
        """
        [result] = self.__call__([x])
        return result

    def add_metrics(self, *metrics: Callable[[list[TData]], list[BaseSolutionQualityInfo]]):
        """
        Adds more metrics to an aggregating solution quality factory.
        :param metrics: New metrics.
        """
        if not self._is_aggregated:
            raise Exception('Cannot add metrics to not aggregating factory.')

        self._metrics.extend(metrics)

    def aggregate(self, *metrics: Callable[[list[TData]], list[BaseSolutionQualityInfo]],
                  metrics_aggregation: Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]],
                                                Iterable[BaseAggregatedSolutionQualityInfo]]):
        """
        Makes a non-aggregating solution quality factory an aggregating one.
        Adds more metrics and their aggregation.
        :param metrics: New metrics to add.
        :param metrics_aggregation: Aggregation for metrics.
        """
        if self._is_aggregated:
            raise Exception('Cannot add aggregation to an already aggregated factory.')
        if not metrics:
            raise Exception('Should add at least one metric for aggregation.')
        self._is_aggregated = True
        self._metrics_aggregation = metrics_aggregation
        self._metrics.extend(metrics)

    def wrap_aggregate(self, *metrics: Callable[[list[TData]], list[BaseSolutionQualityInfo]],
                       metrics_aggregation: Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]],
                                                     Iterable[BaseAggregatedSolutionQualityInfo]]):
        """
        Wraps old metrics and aggregations of an aggregating solution factory as a single metric
        and adds new metrics along with the old "product" (aggregated) metric and an aggregation for them.
        :param metrics: New metrics to be accounted along with the old aggregated metric.
        :param metrics_aggregation: New aggregation to aggregate new metrics along with old aggregated metric.
        """
        if not self._is_aggregated:
            raise Exception('Cannot wrap an non-aggregating solution quality factory. Use `aggregate` method instead.')
        self._metrics = [partial(self._apply_aggregation_static, self._metrics, self._metrics_aggregation),
                         *metrics]
        self._metrics_aggregation = metrics_aggregation

    def _apply_aggregation(self, x: list[TData]) -> Iterable[BaseSolutionQualityInfo]:
        return self._apply_aggregation_static(x, self._metrics, self._metrics_aggregation)

    @staticmethod
    def _apply_aggregation_static(x: list[TData],
                                  metrics: Iterable[Callable[[list[TData]], list[BaseSolutionQualityInfo]]],
                                  metrics_aggregation: Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]],
                                                                Iterable[BaseAggregatedSolutionQualityInfo]]):
        return metrics_aggregation(zip(*[m(x) for m in metrics]))

    def _apply_single_metric(self, x: list[TData]) -> Iterable[BaseSolutionQualityInfo]:
        [metric] = self._metrics
        return metric(x)
