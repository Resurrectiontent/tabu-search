from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable, Hashable, TypeVar

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

    @abstractmethod
    def mutate(self, pivot: Solution) -> List[Solution]:
        ...

    @property
    @abstractmethod
    def _mutation_type(self) -> str:
        ...

    def _mutation_name(self, *args) -> str:
        return self._mutation_type + f'({", ".join(args)})'

    def _one_solution_suffix(*args) -> str:
        return ','.join(map(str, args))
