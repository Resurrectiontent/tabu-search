from abc import ABC, abstractmethod

from tabusearch.solution.base import Solution


class BaseMemoryCriterion(ABC):
    @abstractmethod
    def memorize(self, move: Solution):
        """
        Memories selected move to account it in memory criterion logics. Also, makes a "tick".
        :param move: The selected solution to memorize.
        """

    # @abstractmethod
    # def apply(self, x: Iterable[Solution]) -> Iterable[Solution]:
    #     """
    #     Somehow processes solutions, i.e., "applies" the MemoryCriterion.
    #     :param x: Initial set (collection) of solutions.
    #     :return: Processed collection of solutions. Solutions can be modified or filtered.
    #     """


