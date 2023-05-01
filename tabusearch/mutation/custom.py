from typing import Callable, Iterable

from tabusearch.mutation.base import MutationBehaviour
from tabusearch.typing_ import TData


class CustomMutation(MutationBehaviour):
    """
    Usage:
    ```
    custom_mutation = CustomMutation(quality_func, mutation_func, mutation_type_name)
    mutants = custom_mutation.mutate(pivot_solution)
    ```
    """
    _generate_mutations = None

    def __init__(self, mutation: Callable[[TData], list[tuple[TData, str]]],
                 mutation_type: str):
        self._generate_mutations = mutation
        super().__init__(mutation_type)


def create_custom_mutation(name: str, mutation: Callable[[TData], list[tuple[TData, str]]]) \
        -> CustomMutation:
    return CustomMutation(mutation=mutation, mutation_type=name)
