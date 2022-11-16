from abc import ABC
from enum import Enum, auto
from operator import gt, ge
from typing import Iterable, Set, Callable, Optional

from numpy import NAN

from memory.base import MemoryCriterion
from solution.base import Solution
from solution.id import SolutionId
from solution.quality.base import BaseSolutionQualityInfo

# TODO: introduce library of solution aspiration getters and a convenient way to pass them to AspirationCriterion ctor


class AspirationBoundType(Enum):
    Greater = auto()
    GreaterEquals = auto()


class AspirationCriterion(MemoryCriterion, ABC):
    _aspiration_bound: BaseSolutionQualityInfo
    _aspiration_comparison: Callable[[BaseSolutionQualityInfo, BaseSolutionQualityInfo], bool]

    def __init__(self, bound_type: Optional[AspirationBoundType] = None):
        super().__init__()

        self._aspiration_comparison = gt if bound_type is AspirationBoundType.Greater else ge
        self._aspiration_bound = NAN

    def _criterion(self, x: Iterable[Solution]) -> Set[SolutionId]:
        return {s.id for s in x} \
            if self._aspiration_bound is NAN \
            else {s.id
                  for s in x
                  if self._aspiration_comparison(s.quality, self._aspiration_bound)}

    def memorize(self, move: Solution):
        if self._aspiration_bound is NAN or self._aspiration_comparison(move.quality, self._aspiration_bound):
            self._aspiration_bound = move.quality
