from functools import partial
from operator import add, sub
from typing import Iterable, Tuple, Callable, TypeVar, List

from numpy import ndarray

from mutation.base import BidirectionalMutationBehaviour


TElem = TypeVar('TElem', bounds=[int, float])


class NearestNeighboursMutation(BidirectionalMutationBehaviour):
    _mutation_type = 'NN'

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

    def _generate_one_direction_mutations(self, x: ndarray, left: bool) -> List[Tuple[str, ndarray]]:
        inc = partial(add, b=1)
        dec = partial(add, b=-1)

        op = dec if left else inc

        r = []
        for i, el in enumerate(x):
            x = x.copy()
            x[i] = op(el)
            r.append((self._one_solution_suffix('-' if left else '+', str(i)), x))
        return r


class FullAxisShiftMutation(BidirectionalMutationBehaviour):
    _mutation_type = 'FullShift'

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

    def _generate_mutations(self, x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        # TODO: shift all elements in x in one or two directions
        pass
