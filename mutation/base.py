from abc import ABC, abstractmethod
from typing import List, Iterable, Tuple

from numpy import ndarray

from solution.base import Solution
from solution.factory import SolutionFactory


class MutationBehaviour(ABC):
    def __init__(self, general_solution_factory: SolutionFactory):
        self._solution_factory = general_solution_factory.for_solution_generator(self._mutation_type)

    def mutate(self, pivot: Solution) -> List[Solution]:
        """
        Main interface for generation of new solution space.
        :param pivot: Previous solution, whom neighbourhood should be found.
        :return: New solution space.
        """
        return [self._solution_factory(*mutation) for mutation in self._generate_mutations(pivot.position)]

    @property
    @abstractmethod
    def _mutation_type(self) -> str:
        """
        Name of mutation. Used to account previous moves.
        """
        ...

    @abstractmethod
    def _generate_mutations(self, x: ndarray) -> Iterable[Tuple[ndarray, str]]:
        """
        Returns all possible mutations of 1D array
        :param x: 1D numpy ndarray
        :return: Collection of all possible mutation positions in tuples with their str name components
        """
        ...
