from _operator import itemgetter
from typing import Callable, Iterable, List

from sortedcontainers import SortedList

from solution.base import Solution


class SolutionSelection:
    _indices_selector: Callable[[int], Iterable[int]]

    def __init__(self, indices_selector: Callable[[int], Iterable[int]] = None):
        self._indices_selector = indices_selector if indices_selector is not None else range

    def select_n_solutions(self, solutions: SortedList[Solution], n: int) -> List[Solution]:
        return itemgetter(*self._indices_selector(n))(solutions)
