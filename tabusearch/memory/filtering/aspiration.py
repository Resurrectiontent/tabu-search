from abc import ABC
from enum import Enum, auto
from operator import gt, ge, attrgetter
from typing import Iterable, Set, Callable, Optional

from numpy import NAN

from tabusearch.memory.filtering.base import FilteringMemoryCriterion
from tabusearch.solution.base import Solution
from tabusearch.solution.id import SolutionId
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.solution.quality.factory import SolutionQualityFactory


# TODO: introduce library of solution aspiration getters and a convenient way to pass them to AspirationCriterion ctor


class AspirationBoundType(Enum):
    Greater = auto()
    GreaterEquals = auto()


class AspirationCriterion(FilteringMemoryCriterion, ABC):
    _aspiration_bound: BaseSolutionQualityInfo

    _aspiration_comparison: Callable[[BaseSolutionQualityInfo, BaseSolutionQualityInfo], bool]
    # TODO: consider calculating _get_solution_aspiration only for solutions dropped by tabu
    _get_solution_aspiration: Callable[[Solution], BaseSolutionQualityInfo]

    def __init__(self, bound_type: Optional[AspirationBoundType] = None,
                 move_aspiration: Optional[SolutionQualityFactory] = None):
        super().__init__()

        self._aspiration_comparison = ge if bound_type is AspirationBoundType.GreaterEquals else gt
        self._get_solution_aspiration = attrgetter('quality') if move_aspiration is None else move_aspiration
        self._aspiration_bound = NAN

    def memorize(self, move: Solution):
        if self._aspiration_bound is NAN or self._is_aspired(move):
            self._aspiration_bound = move.quality

    def _criterion(self, x: Iterable[Solution]) -> Set[SolutionId]:
        return {s.id for s in x} \
            if self._aspiration_bound is NAN \
            else {s.id for s in x if self._is_aspired(s)}

    def _is_aspired(self, new: Solution):
        return self._aspiration_comparison(self._get_solution_aspiration(new), self._aspiration_bound)
