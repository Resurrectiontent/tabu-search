from functools import partial
from operator import add
from typing import Iterable, Tuple, Callable, List

from numpy import ndarray

from mutation.base import BidirectionalMutationBehaviour


class NearestNeighboursMutation(BidirectionalMutationBehaviour):
    _mutation_type = 'NN'

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

    def _generate_one_direction_mutations(self, x: ndarray, negative: bool) -> List[Tuple[str, ndarray]]:
        inc = partial(add, 1)
        dec = partial(add, -1)

        op = dec if negative else inc

        r = []
        for i, el in enumerate(x):
            x = x.copy()
            x[i] = op(el)
            r.append((self._one_solution_suffix('-' if negative else '+', str(i)), x))
        return r


class FullAxisShiftMutation(BidirectionalMutationBehaviour):
    _mutation_type = 'FullShift'

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

    def _generate_one_direction_mutation(self, x: ndarray, negative: bool) -> Tuple[str, ndarray]:
        inc = partial(add, 1)
        dec = partial(add, -1)

        op = dec if negative else inc

        return self._one_solution_suffix('-' if negative else '+'), op(x)

# TODO: implement PivotOppositeShiftMutation,
#  when all elements before some index are shifted one side,
#  and all elements after shifted opposite side.
#  [0, 0, 0, 0, 0] -> PivotOppositeShiftMutation(2) -> [-1, -1, 0, 1, 1].
#  Consider shifting pivot point also.
