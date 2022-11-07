from typing import Iterable, Set, List, Callable, Dict

from memory.base import MemoryCriterion
from mutation.base import Solution, TMoveId


class TabuList(MemoryCriterion):

    _timers: Dict[TMoveId, int]
    _tabu_time_getter:  Callable[[Solution], int]

    def __init__(self, solution_id_getter: Callable[[Solution], TMoveId], tabu_time_getter: Callable[[Solution], int]):
        super().__init__(solution_id_getter)
        self._tabu_time_getter = tabu_time_getter

    def _criterion(self, x: Iterable[Solution]) -> Set[TMoveId]:
        pass

    def _memorize(self, move: Solution):
        # TODO: test
        for k, v in list(self._timers.items()):
            if v == 1:
                del self._timers[k]
            else:
                self._timers[k] -= 1
        self._timers[self._solution_id_getter(move)] = self._tabu_time_getter(move)
