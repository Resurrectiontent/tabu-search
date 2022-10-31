from typing import Iterable, Tuple

from numpy import ndarray

from mutation.base import MutationBehaviour, BidirectionalMutationBehaviour


class Swap2Mutation(MutationBehaviour):
    _mutation_type = 'Swap2'

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


class Swap3Mutation(BidirectionalMutationBehaviour):
    _mutation_type = 'Swap3Single'

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
        if self._negative_direction or self._negative_direction is None:
            container.append(self._generate_one_permute(x, idx, values, True))
        if not self._negative_direction:
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
