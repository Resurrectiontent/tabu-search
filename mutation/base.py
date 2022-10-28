from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable, Hashable, TypeVar, Iterable, Tuple

from numpy import ndarray

TMoveId = TypeVar('TMoveId', bound=Hashable)


@dataclass
class Solution:
    name: str
    position: ndarray
    quality: float


class MutationBehaviour(ABC):
    def __init__(self, quality: Callable[[ndarray], float]):
        self.quality = quality

    def mutate(self, pivot: Solution) -> List[Solution]:
        return [Solution(self._mutation_name(name) if name else self._mutation_name(),
                         solution,
                         self.quality(solution))
                for name, solution in self._generate_mutations(pivot.position)]

    @property
    @abstractmethod
    def _mutation_type(self) -> str:
        ...

    @abstractmethod
    def _generate_mutations(self, x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        """
        Returns all possible mutations of 1D array
        :param x: 1D numpy ndarray
        :return: Collection of all possible mutations without quality
        """
        ...

    def _mutation_name(self, *args) -> str:
        return self._mutation_type + f'({", ".join(args)})'

    def _one_solution_suffix(*args) -> str:
        return ','.join(map(str, args))
