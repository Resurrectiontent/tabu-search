from abc import ABC
from itertools import chain
from typing import Iterable, Callable

from numpy import ndarray

from convergence.base import ConvergenceCriterion
from memory.base import MemoryCriterion
from mutation.base import Solution, MutationBehaviour


# TODO: [1] Introduce SolutionFactory to avoid solution_id_getter's

# TODO: [2] Consider introducing separate class for quality instead of float


class TabuSearch(ABC):
    # TODO: store best solution. Consider several best, if needed.
    # TODO: consider storing current solution
    # TODO: introduce one cumulative memory criterion
    convergence_criterion: ConvergenceCriterion
    mutation_behaviour: Iterable[MutationBehaviour]
    quality: Callable[[ndarray], float]
    _aspiration: MemoryCriterion
    _tabu: MemoryCriterion

    def optimize(self, x0: ndarray):
        # TODO: convert x0 to solution
        x: Solution = x0

        while True:
            neighbours = self.get_neighbours(x)
            x, move = self.choose(neighbours)
            self.memorize_move(move)

            if self.converged():
                break

        return x

    def get_neighbours(self, x: Solution) -> Iterable[Solution]:
        possible_moves = chain(*[behaviour.mutate(x) for behaviour in self.mutation_behaviour])
        # TODO: use one cumulative memory criterion: account tabu-list, momentum (mid-term) and aspiration memory
        #  for getting all possible (best) members
        ...

    def choose(self, neighbours) -> tuple:
        # TODO: sort by quality, probabilistically (statistically) make a choice
        ...

    def memorize_move(self, move: Solution):
        # TODO: update memory
        #  to drop old and store new into the tabu-list, scale momentum and update aspiration.
        #  Also, update best solution, metric and id
        self._tabu.memorize(move)
        self._aspiration.memorize(move)

    def converged(self):
        # TODO: pass the proper argument to convergence criterion
        return self.convergence_criterion.converged()
