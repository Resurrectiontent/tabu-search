from typing import Iterable, Callable, Tuple, Optional

from numpy import ndarray

from mutation.base import MutationBehaviour, Solution


# TODO: make CustomMutation
#  move to mutation/base.py
class CustomPermutation(MutationBehaviour):
    _mutation_type = None
    _generate_mutations = None

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

        # decorate mutation
        self.mutate = self.check_permutation_init(self.mutate)

    @staticmethod
    def check_permutation_init(mutation):
        def check_mutate(self, pivot: Solution):
            assert self._generate_mutations, 'Should initialize permutation function and permutation type name ' \
                                  'by calling class instance with custom permutation. ' \
                                  'E.g., permut = CustomPermutation(quality)(permutation, permutation_type)'
            return mutation(self, pivot)

        return check_mutate

    def __call__(self, permutation: Callable[[ndarray], Iterable[Tuple[str, ndarray]]], permutation_type: str):
        self._permute = permutation
        self.mutation_type = permutation_type


class Swap2Mutation(MutationBehaviour):
    _mutation_type = 'Swap2'

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

    def _generate_mutations(self, x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        r = []
        for i in range(len(x) - 1):
            for j in range(i + 1, len(x)):
                if x[i] == x[j]:
                    continue

                _x = x.copy()
                _x[[i, j]] = _x[[j, i]]
                r.append((self._one_solution_suffix(i, j), _x))

        return r


class Swap3Mutation(MutationBehaviour):
    _mutation_type = 'Swap3Single'
    _left_direction: Optional[bool] = False

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

    def to_left_direction(self):
        self._left_direction = True
        return self

    def to_right_direction(self):
        self._left_direction = False
        return self

    def to_bidirectional(self):
        self._left_direction = None
        return self

    def reverse_direction(self):
        self._left_direction = not self._left_direction if self._left_direction is not None else None
        return self

    def _generate_mutations(self, x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        r = []
        for i in range(len(x) - 2):
            for j in range(i + 1, len(x) - 1):
                for k in range(j + 1, len(x)):
                    xi, xj, xk = x[[i, j, k]]
                    if xi != xj != xk != xi:
                        self._add_permute(r, x, [i, j, k], [xi, xj, xk])

        return r

    def _add_permute(self, container, x, idx, values):
        if self._left_direction or self._left_direction is None:
            container.append(self._generate_one_permute(x, idx, values, True))
        if not self._left_direction:
            container.append(self._generate_one_permute(x, idx, values, False))

    def _generate_one_permute(self, x, idx, values, left: bool) -> Tuple[str, ndarray]:
        values = self._rotate(values, left)
        x = x.copy()
        x[idx] = values
        n = self._one_solution_suffix('l' if left else 'r', *idx)
        return n, x

    @staticmethod
    def _rotate(lst: list, direction_left: bool):
        if direction_left:
            return lst[1:] + [lst[0]]
        else:
            return [lst[-1]] + lst[:-1]

# TODO: implement slice shift
