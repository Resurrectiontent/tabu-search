from typing import Iterable, Set, List, Callable

from memory.base import MemoryCriterion
from mutation.base import Solution, TMoveId


class TabuList(MemoryCriterion):

    _timers: List[int]
    _tabu_time_getter:  Callable[[Solution], int]

    def __init__(self, solution_id_getter: Callable[[Solution], TMoveId], tabu_time_getter: Callable[[Solution], int]):
        super().__init__(solution_id_getter)
        self._tabu_time_getter = tabu_time_getter

    def _criterion(self, x: Iterable[Solution]) -> Set[TMoveId]:
        pass

    def _memorize(self, move: Solution):
        pass

    ...
