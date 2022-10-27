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

    def mutation_name(self, *args) -> str:
        return self.mutation_type + f'({", ".join(args)})'

    @property
    @abstractmethod
    def mutation_type(self) -> str:
        ...

    @abstractmethod
    def mutate(self, pivot: Solution) -> List[Solution]:
        ...
