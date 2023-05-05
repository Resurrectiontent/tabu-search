from typing import Callable, Iterable, Generic

from tabusearch.solution.quality.aggregated import BaseAggregatedSolutionQualityInfo
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.typing_ import TData


class SolutionQualityFactory(Generic[TData]):
    _evaluation_layers: list[tuple[
        list[Callable[[list[TData]], list[BaseSolutionQualityInfo]]],
        Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]], Iterable[BaseAggregatedSolutionQualityInfo]] | None]]

    def __init__(self, *metrics: Callable[[list[TData]], list[BaseSolutionQualityInfo]],
                 metrics_aggregation: Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]],
                                               Iterable[BaseAggregatedSolutionQualityInfo]] | None = None):
        if not metrics_aggregation and len(metrics) > 1:
            raise Exception('SolutionQualityFactory should be initialised either with one metric '
                            'or with several metrics and an aggregation.')

        self._evaluation_layers = [(list(metrics), metrics_aggregation)]

    def __call__(self, x: list[TData]) \
            -> Iterable[BaseSolutionQualityInfo]:
        evaluated: Iterable[BaseSolutionQualityInfo] | None = None

        for metrics, aggregation in self._evaluation_layers:
            evaluated = self._apply_evaluation(x, metrics, aggregation, evaluated)

        return evaluated

    def single(self, x: TData) -> BaseSolutionQualityInfo:
        """
        Generates quality metric for single data instance.
        :param x: Single data instance.
        :return: Solution quality for data instance.
        """
        [result] = self.__call__([x])
        return result

    def add_evaluation_layer(self, *metrics: Callable[[list[tuple[TData, BaseSolutionQualityInfo]]],
                                                      list[BaseSolutionQualityInfo]],
                             metrics_aggregation: Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]],
                                                           Iterable[BaseAggregatedSolutionQualityInfo]]):
        """
        Wraps old metrics and aggregations of an aggregating solution factory as a single metric
        and adds new metrics along with the old "product" (aggregated) metric and an aggregation for them.
        :param metrics: New metrics to be accounted along with the old aggregated metric.
        :param metrics_aggregation: New aggregation to aggregate new metrics along with old aggregated metric.
        """
        if not metrics_aggregation:
            raise Exception('Can add evaluation layer only with aggregation.')
        self._evaluation_layers.append((list(metrics), metrics_aggregation))

    @staticmethod
    def _apply_evaluation(x: list[TData],
                          metrics: Iterable[Callable[[list[TData | tuple[TData, BaseSolutionQualityInfo]]],
                                                     list[BaseSolutionQualityInfo]]],
                          metrics_aggregation: Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]],
                                                        Iterable[BaseAggregatedSolutionQualityInfo]],
                          pre_evaluated_metric: Iterable[BaseSolutionQualityInfo] | None) \
            -> Iterable[BaseSolutionQualityInfo]:
        if not metrics_aggregation:
            [metric, *other_metrics] = metrics
            if pre_evaluated_metric or other_metrics:
                raise Exception('Cannot account more than one metric without aggregation.')
            return metric(x)

        if pre_evaluated_metric:
            # TODO: consider replacing list(zip(x, pre_evaluated_metric)) on just zip(x, pre_evaluated_metric)
            # first - pre-evaluated metric, then - other metrics with passing pre-evaluated value
            # zip(*...) - for recombination of metric[solution] on solution[metric]
            evaluated_metrics = zip(*([pre_evaluated_metric]
                                      + [m(list(zip(x, pre_evaluated_metric))) for m in metrics]))
        else:
            evaluated_metrics = zip(*[m(x) for m in metrics])

        return metrics_aggregation(evaluated_metrics)
