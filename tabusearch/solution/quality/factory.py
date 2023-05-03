from typing import Callable, Iterable, Generic

from tabusearch.solution.quality.aggregated import BaseAggregatedSolutionQualityInfo
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.typing_ import TData


class SolutionQualityFactory(Generic[TData]):
    _factory: Callable[[list[TData]], Iterable[BaseSolutionQualityInfo]]

    # TODO: consider moving ctor logics to helper functions
    #  and taking final `(ndarray) -> BaseSolutionQualityInfo` callable as an __init__ argument
    def __init__(self, *metrics: Callable[[list[TData]], list[BaseSolutionQualityInfo]],
                 metrics_aggregation: Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]],
                                               Iterable[BaseAggregatedSolutionQualityInfo]]
                                      | None = None):
        assert len(metrics) == 1 or metrics_aggregation is not None and len(metrics) > 1

        match metrics:
            case [single_metric]:
                self._factory = single_metric
            case _:
                def factory(x: list[TData]) -> Iterable[BaseSolutionQualityInfo]:
                    return metrics_aggregation(zip(*[m(x) for m in metrics]))
                self._factory = factory

    def __call__(self, x: list[TData]) -> Iterable[BaseSolutionQualityInfo]:
        return self._factory(x)

    def single(self, x: TData) -> BaseSolutionQualityInfo:
        result, *_ = self._factory([x])
        return result
