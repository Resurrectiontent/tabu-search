from functools import partial

from typing import Callable, Iterable, Tuple

from tabusearch.mutation.base import MutationBehaviour
from tabusearch.solution.factory import SolutionFactory
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
    _mutation_type = None

    def __init__(self, general_solution_factory: SolutionFactory,
                 mutation: Callable[[TData], Iterable[Tuple[TData, str]]],
                 mutation_type: str):
        # should set mutation type before superclass initialisation
        self._generate_mutations = mutation
        self._mutation_type = mutation_type

        super().__init__(general_solution_factory)


def create_custom_mutation(name: str, mutation: Callable[[TData], list[tuple[TData, str]]]) \
        -> Callable[[SolutionFactory], CustomMutation]:
    return partial(CustomMutation, mutation=mutation, mutation_type=name)
