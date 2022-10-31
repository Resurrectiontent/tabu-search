from typing import Iterable, Tuple, Callable

from numpy import ndarray

from mutation.base import BidirectionalMutationBehaviour


class NearestNeighboursMutation(BidirectionalMutationBehaviour):
    _mutation_type = 'NN'

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

    # TODO: finish implementation
    def _generate_mutations(self, x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        r = []
        for i in range(len(x)):
            ...
