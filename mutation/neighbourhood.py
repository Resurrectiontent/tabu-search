from typing import Iterable, Tuple, Callable

from numpy import ndarray

from mutation.base import MutationBehaviour


class NearestNeighboursMutation(MutationBehaviour):
    _mutation_type = 'NN'
    _negative = False

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

    # TODO: move change direction behaviour to separate class
    def to_negative_direction(self):
        self._negative = True
        return self

    def to_positive_direction(self):
        self._negative = False
        return self

    def to_bidirectional(self):
        self._negative = None
        return self

    def reverse_direction(self):
        self._negative = not self._negative if self._negative is not None else None
        return self

    # TODO: finish implementation
    def _generate_mutations(self, x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        r = []
        for i in range(len(x)):
            ...
