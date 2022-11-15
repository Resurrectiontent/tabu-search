from functools import partial
from typing import Callable

from solution.base import Solution
from solution.id import SolutionId


class SolutionFactory:
    def __init__(self, position_quality_getter: Callable):
        self._initialized = False
        self._solution_id_factory = None
        # TODO: initialize quality factory
        ...

    # TODO: impement factory call

    def for_solution_generator(self, generator_name: str):
        self._solution_id_factory = partial(SolutionId, generator_name)
        self._initialized = True
