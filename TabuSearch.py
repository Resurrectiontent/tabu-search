from abc import ABC
from typing import Generic, TypeVar

import numpy as np
from numpy import ndarray

# TODO: Consider designing separate class for tabu list
TL = TypeVar('TL')  # Tabu list


class TabuSearch(ABC, Generic[TL]):
    _tabu_list: TL

    def optimize(self, x0: ndarray):
        x = x0

        while not self.convergence_criterion.converged(x):
            neighbours = self.get_neighbours(x)
            x, move = self.choose(neighbours)
            self.make_tick()
            self.memorize_move(move)

        return x

