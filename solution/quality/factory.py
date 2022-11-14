from functools import partialmethod
from typing import Callable, Iterable, Optional

from numpy import ndarray

from solution.quality.base import BaseSolutionQualityInfo


class SolutionQualityFactory:
    _factory = Callable[[ndarray], BaseSolutionQualityInfo]

    # TODO: abstract *metrics creation to utility functions
    #  consider introducing SolutionQualityMetric or just partial for BaseSolutionQualityInfo-inherited ctors
    def __init__(self, *metrics: Callable[[ndarray], BaseSolutionQualityInfo],
                 metrics_aggregation: Optional[Callable[[Iterable[BaseSolutionQualityInfo]],
                                                        BaseSolutionQualityInfo]] = None):
        assert len(metrics) == 1 or metrics_aggregation is not None and len(metrics) > 1

        if len(metrics) == 1:
            self._factory,  = metrics
        else:
            def factory(x: ndarray):
                return metrics_aggregation([m(x) for m in metrics])
            self._factory = factory

    __call__ = partialmethod(_factory)
