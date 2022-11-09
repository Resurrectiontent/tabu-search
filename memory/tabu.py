from typing import Iterable, Set, List, Callable, Dict

from memory.base import MemoryCriterion
from mutation.base import TMoveId
from solution.base import Solution


# TODO: introduce library of tabu time getters and a convenient way to pass them to TabuList ctor


class TabuList(MemoryCriterion):
    """
    Represents short-term memory in Tabu Search algorithm (tabu list).
    """

    _timers: Dict[TMoveId, int]
    _tabu_time_getter:  Callable[[Solution], int]

    def __init__(self, solution_id_getter: Callable[[Solution], TMoveId], tabu_time_getter: Callable[[Solution], int]):
        super().__init__(solution_id_getter)
        self._tabu_time_getter = tabu_time_getter

    def _criterion(self, x: Iterable[Solution]) -> Set[TMoveId]:
        return {id_ for id_ in [self._solution_id_getter(s) for s in x] if id_ not in self._timers}

    def _memorize(self, move: Solution):
        for k, v in list(self._timers.items()):
            if v == 1:
                del self._timers[k]
            else:
                self._timers[k] -= 1
        self._timers[self._solution_id_getter(move)] = self._tabu_time_getter(move)
