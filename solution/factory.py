from copy import copy
from functools import partial
from typing import Callable, Iterable, Optional

from numpy import ndarray

from solution.base import Solution
from solution.id import SolutionId
from solution.quality.aggregated import BaseAggregatedSolutionQualityInfo
from solution.quality.base import BaseSolutionQualityInfo
from solution.quality.factory import SolutionQualityFactory


class SolutionFactory:

    # TODO: abstract *metrics creation to utility functions
    #  consider partial for BaseSolutionQualityInfo-inherited ctors
    def __init__(self, *metrics: Callable[[ndarray], BaseSolutionQualityInfo],
                 metrics_aggregation: Optional[Callable[[Iterable[BaseSolutionQualityInfo]],
                                                        BaseAggregatedSolutionQualityInfo]] = None):
        self._id_factory = None

        self._quality_factory = SolutionQualityFactory(*metrics, metrics_aggregation)

    @property
    def is_initialized(self):
        return self._id_factory is not None

    def __call__(self, position: ndarray, *name_suffix: str) -> Solution:
        assert self._id_factory, 'SolutionFactory is not initialized.' \
                                 ' Make a solution generator specific factory by calling' \
                                 ' generator_solution_factory = general_solution_factory.' \
                                 'for_solution_generator(generator_name).'
        return Solution(self._id_factory(name_suffix),
                        position,
                        self._quality_factory(position))

    def initial(self, position: ndarray) -> Solution:
        return Solution(SolutionId('Init'), position, self._quality_factory(position))

    def for_solution_generator(self, generator_name: str) -> 'SolutionFactory':
        specialized = copy(self)
        specialized._id_factory = partial(SolutionId, generator_name)
        return specialized

