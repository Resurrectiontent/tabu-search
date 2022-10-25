from abc import ABC
from typing import Generic, TypeVar, Dict, Iterable, Callable, Tuple, List

from numpy import ndarray

from convergence.base import ConvergenceCriterion
from memory.base import MemoryCriterion
from moves.base import Move

# TODO: Consider designing separate class for tabu list
TL = TypeVar('TL')  # Tabu list


class TabuSearch(ABC, Generic[TL]):
    convergence_criterion: ConvergenceCriterion
    possible_moves: Dict[str, Callable[[ndarray], Iterable[ndarray]]]
    quality: Callable[[ndarray], float]
    _aspiration: MemoryCriterion
    _tabu: MemoryCriterion

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

    def get_neighbours(self, x) -> Dict[str, List[Tuple[ndarray, ndarray]]]:
        possible_moves = {move: func(x) for move, func in self.possible_moves.items()}
        # TODO: consider calculating quality for aspiration
        # TODO: account tabu-list and aspiration memory
        ...

    def choose(self, neighbours) -> tuple:
        # TODO: sort by quality, probabilistically (statistically) make a choice
        ...

    def make_tick(self):
        # TODO: pop tabu-list
        ...

    def memorize_move(self, move: Move):
        # TODO: push tabu-list, mid-term memory
        self._tabu.memorize(move)
        self._aspiration.memorize(move)

    def converged(self):
        # TODO: update argument
        return self.convergence_criterion.converged()
