from typing import Callable

from solution.base import Solution


class SolutionFactory:
    def __init__(self, solution_id_getter: Callable[[Solution], float], position_quality_getter: Callable):
        ...
