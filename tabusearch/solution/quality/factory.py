from typing import Callable, Iterable, Optional, Generic

from tabusearch.solution.quality.aggregated import BaseAggregatedSolutionQualityInfo
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.typing_ import TData


class SolutionQualityFactory(Generic[TData]):
    _factory: Callable[[TData], BaseSolutionQualityInfo]

    # TODO: consider moving ctor logics to helper functions
    #  and taking final `(ndarray) -> BaseSolutionQualityInfo` callable as an __init__ argument
    def __init__(self, *metrics: Callable[[TData], BaseSolutionQualityInfo],
                 metrics_aggregation: Optional[Callable[[Iterable[BaseSolutionQualityInfo]],
                                                        BaseAggregatedSolutionQualityInfo]] = None):
        assert len(metrics) == 1 or metrics_aggregation is not None and len(metrics) > 1

        if len(metrics) == 1:
            self._factory,  = metrics
        else:
            def factory(x: TData):
                return metrics_aggregation([m(x) for m in metrics])
            self._factory = factory

    def __call__(self, x: TData) -> BaseSolutionQualityInfo:
        return self._factory(x)
