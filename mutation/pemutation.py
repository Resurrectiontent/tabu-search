from abc import ABC, abstractmethod
from typing import Iterable, List, Callable, Dict, Tuple

from numpy import ndarray

from mutation.base import MutationBehaviour, Solution


# TODO: Move to some base class
def single_solution_suffix(*args) -> str:
    return ','.join(args)


# TODO: Do we need this?
class PermutationMutation(MutationBehaviour, ABC):
    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

    def mutate(self, pivot: Solution) -> List[Solution]:
        return [Solution(self.mutation_name(name) if name else self.mutation_name(),
                         solution,
                         self.quality(solution))
                for name, solution in self._permute(pivot.position)]

    @staticmethod
    @abstractmethod
    def _permute(x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        """
        Returns all possible permutations of 1D array
        :param x: 1D numpy ndarray
        :return: Collection of all possible permutations without quality
        """
        ...


class CustomPermutation(PermutationMutation):
    mutation_type = None
    _permute = None

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

        # decorate mutation
        self.mutate = self.check_permutation_init(self.mutate)

    @staticmethod
    def check_permutation_init(mutation):
        def check_mutate(self, pivot: Solution):
            assert self._permute, 'Should initialize permutation function and permutation type name ' \
                                  'by calling class instance with custom permutation. ' \
                                  'E.g., permut = CustomPermutation(quality)(permutation, permutation_type)'
            return mutation(self, pivot)

        return check_mutate

    def __call__(self, permutation: Callable[[ndarray], Iterable[Tuple[str, ndarray]]], permutation_type: str):
        self._permute = permutation
        self.mutation_type = permutation_type


class Swap2Mutation(PermutationMutation):
    mutation_type = 'Swap2'

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

    @staticmethod
    def _permute(x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        r = []
        for i in range(len(x) - 1):
            for j in range(i + 1, len(x)):
                if x[i] == x[j]:
                    continue

                _x = x.copy()
                _x[[i, j]] = _x[[j, i]]
                r.append((single_solution_suffix(i, j), _x))

        return r


class Swap3SingleDirectionMutation(PermutationMutation):

    left_direction = False

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

    @property
    def mutation_type(self) -> str:
        return f'Swap3Single'

    def to_left_direction(self):
        self.left_direction = True
        return self

    def to_right_direction(self):
        self.left_direction = False
        return self

    def reverse_direction(self):
        self.left_direction = not self.left_direction
        return self

    @staticmethod
    def _permute(x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        # TODO: finish implementation
        #  resolve left_direction field reference necessity in staticmethod
        r = []
        for i in range(len(x) - 2):
            for j in range(i + 1, len(x) - 1):
                for k in range(j + 1, len(x)):
                    ...
