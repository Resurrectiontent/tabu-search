from typing import Callable, Iterable, Optional

from numpy import ndarray

from solution.quality.base import BaseSolutionQualityInfo


class SolutionQualityFactory:
    # TODO: abstract *metrics creation to utility functions
    #  consider introducing SolutionQualityMetric or just partial for BaseSolutionQualityInfo-inherited ctors
    def __init__(self, *metrics: Callable[[ndarray], BaseSolutionQualityInfo],
                 metrics_aggregation: Optional[Callable[[Iterable[BaseSolutionQualityInfo]],
                                                        BaseSolutionQualityInfo]] = None):
        assert len(metrics) == 1 or metrics_aggregation is not None
        # TODO: finish constructor
        ...
