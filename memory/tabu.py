from typing import Iterable, Set, List, Callable, Dict

from memory.base import MemoryCriterion
from solution.base import Solution
from solution.id import SolutionId


# TODO: introduce library of tabu time getters and a convenient way to pass them to TabuList ctor


class TabuList(MemoryCriterion):
    """
    Represents short-term memory in Tabu Search algorithm (tabu list).
    """

    _timers: Dict[SolutionId, int]
    _tabu_time_getter:  Callable[[Solution], int]

    def __init__(self, tabu_time_getter: Callable[[Solution], int]):
        super().__init__()
        self._tabu_time_getter = tabu_time_getter

    def _criterion(self, x: Iterable[Solution]) -> Set[SolutionId]:
        return {id_ for id_ in [s.id for s in x] if id_ not in self._timers}

    def _memorize(self, move: Solution):
        for k, v in list(self._timers.items()):
            if v == 1:
                del self._timers[k]
            else:
                self._timers[k] -= 1
        self._timers[move.id] = self._tabu_time_getter(move)
