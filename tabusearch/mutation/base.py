from abc import ABC, abstractmethod
from typing import Generic

from tabusearch.solution.base import Solution
from tabusearch.typing_ import TData


class MutationBehaviour(ABC, Generic[TData]):
    _mutation_type: str = None

    def __init__(self, mutation_type: str):
        self._mutation_type = mutation_type

    def mutate(self, pivot: Solution) -> list[tuple[TData, str]]:
        """
        Main interface for generation of new solution space.
        :param pivot: Previous solution, whom neighbourhood should be found.
        :return: New solution space.
        """
        # TODO: erase
        # return [self._solution_factory(*mutation) for mutation in self._generate_mutations(pivot.position)]
        return self._generate_mutations(pivot.position)

    @property
    def mutation_type(self) -> str:
        """
        Name of mutation. Used to account previous moves.
        """
        return self._mutation_type

    @abstractmethod
    def _generate_mutations(self, x: TData) -> list[tuple[TData, str]]:
        """
        Returns all possible mutations of 1D array
        :param x: 1D numpy ndarray
        :return: Collection of all possible mutation positions in tuples with their str name components
        """
        ...
