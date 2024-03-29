from numpy.typing import NDArray

from tabusearch.mutation.base import MutationBehaviour
from tabusearch.mutation.directed import BidirectionalMutationBehaviour

# TODO: consider moving from classes to functions, passed to superclass ctor
#  (as these classes implement only one function). Think twice on Swap3Mutation


class Swap2Mutation(MutationBehaviour[NDArray]):
    """
    Implements mutation behaviour, in which all pairs of (non-equal) elements are permuted in different solutions.
    """
    def __init__(self):
        super().__init__('Swap2')

    def _generate_mutations(self, x: NDArray) -> list[tuple[NDArray, str]]:
        r = []
        for i in range(len(x) - 1):
            for j in range(i + 1, len(x)):
                if x[i] == x[j]:
                    continue

                _x = x.copy()
                _x[[i, j]] = _x[[j, i]]
                r.append((_x, str(i), str(j)))

        # noinspection PyTypeChecker
        return r


class Swap3Mutation(BidirectionalMutationBehaviour[NDArray]):
    """
    Implements mutation behaviour, in which all triplets (without equal elements)
    can be permuted in all two possible ways:
    ```([1,2,3] -> [2,3,1] and/or [3,1,2])``` in different solutions.
    """
    def __init__(self):
        super().__init__('Swap3Single')

    # Override default logics to reduce cycle initializations
    def _generate_mutations(self, x: NDArray) -> list[tuple[NDArray, str]]:
        r = []
        for i in range(len(x) - 2):
            for j in range(i + 1, len(x) - 1):
                for k in range(j + 1, len(x)):
                    xi, xj, xk = x[[i, j, k]]
                    if xi != xj != xk != xi:
                        self._add_permute(r, x, [i, j, k], [xi, xj, xk])

        return r

    def _add_permute(self, container, x, idx, values):
        if self.direction.is_negative:
            container.append(self._generate_one_permute(x, idx, values, True))
        if self.direction.is_positive:
            container.append(self._generate_one_permute(x, idx, values, False))

    def _generate_one_permute(self, x, idx, values, negative: bool) -> tuple[NDArray, str]:
        values = self._rotate(values, negative)
        x = x.copy()
        x[idx] = values
        n = 'l' if negative else 'r'
        # noinspection PyTypeChecker
        return x, n, *list(map(str, idx))

    @staticmethod
    def _rotate(lst: list, direction_left: bool):
        if direction_left:
            return lst[1:] + [lst[0]]
        else:
            return [lst[-1]] + lst[:-1]

# TODO: implement Reposition1Mutation
#  [1,2,3,4,5,6,7,8,9,0] -> Reposition1Mutation(5, 2) -> [0,1,6,3,4,5,7,8,9,0]
