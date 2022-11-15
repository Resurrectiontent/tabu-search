from abc import ABC, abstractmethod
from typing import List, Callable, Hashable, TypeVar, Iterable, Tuple, Optional

from numpy import ndarray

from solution.base import Solution

# TODO: Ensure removing usage and remove
TMoveId = TypeVar('TMoveId', bound=Hashable)


class MutationBehaviour(ABC):
    def __init__(self, quality: Callable[[ndarray], float]):
        self.quality = quality

    def mutate(self, pivot: Solution) -> List[Solution]:
        """
        Main interface for generation of new solution space.
        :param pivot: Previous solution, whom neighbourhood should be found.
        :return: New solution space.
        """
        return [Solution(self._mutation_name(name),
                         solution,
                         self.quality(solution))
                for name, solution in self._generate_mutations(pivot.position)]

    @property
    @abstractmethod
    def _mutation_type(self) -> str:
        """
        Name of mutation. Used to account previous moves.
        """
        ...

    @abstractmethod
    def _generate_mutations(self, x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        """
        Returns all possible mutations of 1D array
        :param x: 1D numpy ndarray
        :return: Collection of all possible mutations without quality
        """
        ...

    def _mutation_name(self, suffix: Optional[str] = None) -> str:
        """
        Generates name of mutation using its type and its custom params.
        :param suffix: Params name, gotten via _one_solution_suffix.
        """
        return f'{self._mutation_type}({suffix})' if suffix else self._mutation_type

    def _one_solution_suffix(*args) -> str:
        """
        Generates suffix for specific solution, given its params.
        :param args: params of a specific solution.
        """
        return ','.join(map(str, args))
