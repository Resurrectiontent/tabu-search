from functools import wraps
from typing import Callable, Iterable, Tuple

from numpy import ndarray

from mutation.base import MutationBehaviour, Solution


class CustomMutation(MutationBehaviour):
    """
    Usage:
    ```
    custom_mutation = CustomMutation(quality_func)(mutation_func, mutation_type_name)
    mutants = custom_mutation.mutate(pivot_solution)
    ```
    """
    _mutation_type = None
    _generate_mutations = None

    def __init__(self, quality: Callable[[ndarray], float]):
        super().__init__(quality)

        # decorate mutation
        self.mutate = self.check_mutation_inited(self.mutate)

    @staticmethod
    def check_mutation_inited(mutation):
        @wraps(mutation)
        def check_mutate(self, pivot: Solution):
            assert self._generate_mutations, 'Should initialize mutation function and mutation type name ' \
                                  'by calling class instance with custom mutation. ' \
                                  'E.g., mut = CustomMutation(quality)(mutation, mutation_type)'
            return mutation(self, pivot)

        return check_mutate

    def __call__(self, mutation: Callable[[ndarray], Iterable[Tuple[str, ndarray]]], mutation_type: str):
        self._generate_mutations_ = mutation
        self.mutation_type = mutation_type
