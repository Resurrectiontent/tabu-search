from typing import Callable, Iterable, Tuple

from numpy import ndarray

from tabusearch.mutation.base import MutationBehaviour
from tabusearch.solution.factory import SolutionFactory


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
                 mutation: Callable[[ndarray], Iterable[Tuple[ndarray, str]]],
                 mutation_type: str):
        super().__init__(general_solution_factory)

        self._generate_mutations = mutation
        self._mutation_type = mutation_type
