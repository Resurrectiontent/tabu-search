from abc import ABC
from enum import Enum, auto
from operator import gt, ge
from typing import Iterable, Set, Callable, Optional

from numpy import NAN

from memory.base import MemoryCriterion
from mutation.base import TMoveId
from solution.base import Solution


# TODO: introduce library of solution aspiration getters and a convenient way to pass them to AspirationCriterion ctor


class AspirationBoundType(Enum):
    Greater = auto()
    GreaterEquals = auto()


class AspirationCriterion(MemoryCriterion, ABC):
    _solution_aspiration_getter: Callable[[Solution], float]
    _aspiration_bound: float
    _aspiration_comparison: Callable[[float, float], bool]

    def __init__(self, solution_id_getter: Callable[[Solution], TMoveId],
                 solution_aspiration_getter: Callable[[Solution], float],
                 bound_type: Optional[AspirationBoundType] = AspirationBoundType.Greater):
        super().__init__(solution_id_getter)

        self._solution_aspiration_getter = solution_aspiration_getter
        self._aspiration_comparison = gt if bound_type is AspirationBoundType.Greater else ge
        self._aspiration_bound = NAN

    def _criterion(self, x: Iterable[Solution]) -> Set[TMoveId]:
        return {s.id for s in x} \
            if self._aspiration_bound is NAN \
            else {s.id
                  for s in x
                  if self._aspiration_comparison(self._solution_aspiration_getter(s), self._aspiration_bound)}

    def _memorize(self, move: Solution):
        move_aspiration = self._solution_aspiration_getter(move)
        if self._aspiration_comparison(move_aspiration, self._aspiration_bound):
            self._aspiration_bound = move_aspiration
