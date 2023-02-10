from typing import Callable, Iterable, Optional

from numpy.typing import NDArray

from solution.quality.aggregated import BaseAggregatedSolutionQualityInfo
from solution.quality.base import BaseSolutionQualityInfo


class SolutionQualityFactory:
    _factory: Callable[[NDArray], BaseSolutionQualityInfo]

    # TODO: consider moving ctor logics to helper functions
    #  and taking final `(ndarray) -> BaseSolutionQualityInfo` callable as an __init__ argument
    def __init__(self, *metrics: Callable[[NDArray], BaseSolutionQualityInfo],
                 metrics_aggregation: Optional[Callable[[Iterable[BaseSolutionQualityInfo]],
                                                        BaseAggregatedSolutionQualityInfo]] = None):
        assert len(metrics) == 1 or metrics_aggregation is not None and len(metrics) > 1

        if len(metrics) == 1:
            self._factory,  = metrics
        else:
            def factory(x: NDArray):
                return metrics_aggregation([m(x) for m in metrics])
            self._factory = factory

    def __call__(self, x: NDArray) -> BaseSolutionQualityInfo:
        return self._factory(x)
