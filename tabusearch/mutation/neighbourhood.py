from functools import partial
from operator import add

from numpy.typing import NDArray

from tabusearch.mutation.directed import BidirectionalMutationBehaviour

# TODO: consider moving from classes to functions, passed to superclass ctor
#  (as these classes implement only one function)


class NearestNeighboursMutation(BidirectionalMutationBehaviour[NDArray]):
    """
    Implements mutation behaviour, in which we add and/or subtract 1 from every element in different solutions
    ```
    [0,0,0] -> [1,0,0], [0,1,0], [0,0,1] and/or [-1,0,0], [0,-1,0], [0,0,-1]
    ```
    """
    def __init__(self):
        super().__init__('NN')

    def _generate_one_direction_mutations(self, x: NDArray, negative: bool) -> list[tuple[NDArray, str, str]]:
        inc = partial(add, 1)
        dec = partial(add, -1)

        op, mark = (dec, '-') if negative else (inc, '+')

        r = []
        for i, el in enumerate(x):
            x_ = x.copy()
            x_[i] = op(el)
            r.append((x_, mark, str(i)))
        return r


class FullAxisShiftMutation(BidirectionalMutationBehaviour[NDArray]):
    """
    Implements mutation behaviour, in which we shift all elements to negative and/or to positive side
    ```
    [0,0,0] -> [1,1,1] and/or [-1,-1,-1]
    ```
    """
    def __init__(self):
        super().__init__('FullShift')

    def _generate_one_direction_mutation(self, x: NDArray, negative: bool) -> tuple[NDArray, str]:
        inc = partial(add, 1)
        dec = partial(add, -1)

        return (dec(x), '-') if negative else (inc(x), '+')

# TODO: implement PivotOppositeShiftMutation,
#  when all elements before some index are shifted one side,
#  and all elements after shifted opposite side.
#  [0, 0, 0, 0, 0] -> PivotOppositeShiftMutation(2) -> [-1, -1, 0, 1, 1].
#  Consider shifting pivot point also.
