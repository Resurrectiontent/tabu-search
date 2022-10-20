from abc import ABC
from typing import Generic, TypeVar, Dict, Iterable, Callable

import numpy as np
from numpy import ndarray

from convergence.base import ConvergenceCriterion

# TODO: Consider designing separate class for tabu list
TL = TypeVar('TL')  # Tabu list


class TabuSearch(ABC, Generic[TL]):
    convergence_criterion: ConvergenceCriterion
    possible_moves: Dict[str, Callable[[ndarray], Iterable[ndarray]]]
    _tabu_list: TL

    def optimize(self, x0: ndarray):
        x = x0

        while True:
            neighbours = self.get_neighbours(x)
            x, move = self.choose(neighbours)
            self.make_tick()
            self.memorize_move(move)

            if self.converged():
                break

        return x

    def get_neighbours(self, x):
        possible_moves = {move: func(x) for move, func in self.possible_moves.items()}
        # TODO: consider calculating quality for aspiration
        # TODO: account tabu-list and aspiration criteria
        ...

    def choose(self, neighbours) -> tuple:
        # TODO: sort by quality, probabilistically (statistically) make a choice
        ...

    def make_tick(self):
        # TODO: pop tabu-list
        ...

    def memorize_move(self, move):
        # TODO: push tabu-list, mid-term memory
        ...

    def converged(self):
        # TODO: update argument
        return self.convergence_criterion.converged()
