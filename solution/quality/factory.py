from typing import Callable, Iterable, Optional

from numpy import ndarray

from solution.quality.aggregated import BaseAggregatedSolutionQualityInfo
from solution.quality.base import BaseSolutionQualityInfo


class SolutionQualityFactory:
    _factory = Callable[[ndarray], BaseSolutionQualityInfo]

    # TODO: consider moving ctor logics to helper functions
    #  and taking final `(ndarray) -> BaseSolutionQualityInfo` callable as an __init__ argument
    def __init__(self, *metrics: Callable[[ndarray], BaseSolutionQualityInfo],
                 metrics_aggregation: Optional[Callable[[Iterable[BaseSolutionQualityInfo]],
                                                        BaseAggregatedSolutionQualityInfo]] = None):
        assert len(metrics) == 1 or metrics_aggregation is not None and len(metrics) > 1

        if len(metrics) == 1:
            self._factory,  = metrics
        else:
            def factory(x: ndarray):
                return metrics_aggregation([m(x) for m in metrics])
            self._factory = factory

    __call__ = _factory
