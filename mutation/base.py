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


class BidirectionalMutationBehaviour(MutationBehaviour, ABC):
    # Override one of these methods in dependence on whether mutation generates
    #  single new instance or multiple new indices. Last tuple param is for *args.
    #  By default, one of them is called in _generate_mutations.
    _generate_one_direction_mutation: Callable[[ndarray, bool], Tuple[str, ndarray]]
    _generate_one_direction_mutations: Callable[[ndarray, bool], List[Tuple[str, ndarray]]]

    _negative_direction = False

    def _generate_mutations(self, x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        """
        Default implementation for bidirectional mutations. Override, if you need to change it.
        """
        assert self._generate_one_direction_mutation or self._generate_one_direction_mutations, \
            'Should implement either _generate_one_direction_mutation or _generate_one_direction_mutations methods' \
            ' to use default logics or override _generate_mutations to implement custom logics.'
        r = []

        recorder, generator = (r.extend, self._generate_one_direction_mutations) \
            if self._generate_one_direction_mutations \
            else (r.append, self._generate_one_direction_mutation)

        if self._negative_direction or self._negative_direction is None:
            recorder(generator(x, True))
        if not self._negative_direction:
            recorder(generator(x, False))

        return r

    def to_negative_direction(self):
        self._negative_direction = True
        return self

    def to_positive_direction(self):
        self._negative_direction = False
        return self

    def to_bidirectional(self):
        self._negative_direction = None
        return self

    def reverse_direction(self):
        self._negative_direction = not self._negative_direction if self._negative_direction is not None else None
        return self
