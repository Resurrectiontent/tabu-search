from _operator import itemgetter
from typing import Callable

from sortedcontainers import SortedList

from solution.base import Solution


class SolutionSelection:
    _idx_selector: Callable[[int], int]

    def __init__(self, idx_selector: Callable[[int], int] = None):
        """
        Initializes solution selection strategy.
        :param idx_selector: A function to select solution from the best.
        It takes length of solution space and returns an index of solution to select
        (0 - actual best with higher quality, 1 - second best, etc.).
        Defaults to selection of the best solution.
        """
        self._idx_selector = idx_selector if idx_selector is not None else lambda: 0

    def __call__(self, solutions: SortedList[Solution]) -> Solution:
        """
        Selects the "best" solution from ordered `solutions`. "Best" means defined by solution selection strategy.
        :param solutions: `SortedList` of `Solution` objects ordered by ascending quality (best - last).
        :return: The "best" solution in the collection.
        """
        return itemgetter(self._idx_selector(len(solutions)))(list(reversed(solutions)))
