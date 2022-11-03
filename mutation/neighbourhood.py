from functools import partial
from operator import add
from typing import Tuple, List

from numpy import ndarray

from mutation.directed import BidirectionalMutationBehaviour


class NearestNeighboursMutation(BidirectionalMutationBehaviour):
    """
    Implements mutation behaviour, in which we add and/or subtract 1 from every element in different solutions
    ```
    [0,0,0] -> [1,0,0], [0,1,0], [0,0,1] and/or [-1,0,0], [0,-1,0], [0,0,-1]
    ```
    """
    _mutation_type = 'NN'

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
    """
    Implements mutation behaviour, in which we shift all elements to negative and/or to positive side
    ```
    [0,0,0] -> [1,1,1] and/or [-1,-1,-1]
    ```
    """
    _mutation_type = 'FullShift'

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
