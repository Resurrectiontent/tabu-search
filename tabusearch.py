from abc import ABC
from itertools import chain
from typing import Iterable
from numpy import ndarray
from sortedcontainers import SortedList

from convergence.base import ConvergenceCriterion
from memory.aspiration import AspirationCriterion
from memory.base import BaseMemoryCriterion
from memory.tabu import TabuList
from mutation.base import MutationBehaviour
from solution.base import Solution
from solution.factory import SolutionFactory

# TODO: implement solution selection
# TODO: consider possibility of using different metrics for solution selection and aspiration


class TabuSearch(ABC):
    hall_of_fame: SortedList  # sorted by ascending quality
    convergence_criterion: ConvergenceCriterion
    mutation_behaviour: Iterable[MutationBehaviour]
    solution_factory: SolutionFactory
    aspiration: AspirationCriterion
    tabu: TabuList
    _memory: BaseMemoryCriterion

    @property
    def resulting_memory_criterion(self):
        if self._memory is None:
            self._memory = self.tabu and self.aspiration
        return self._memory

    def optimize(self, x0: ndarray):
        x = self.solution_factory.initial(x0)

        while True:
            neighbours = self.get_neighbours(x)
            x = self.choose(neighbours)
            self.memorize_move(x)

            if self.converged(x):
                break

        return x

    def get_neighbours(self, x: Solution) -> Iterable[Solution]:
        possible_moves = chain(*[behaviour.mutate(x) for behaviour in self.mutation_behaviour])
        return self.resulting_memory_criterion.filter(possible_moves)

    def choose(self, neighbours) -> Solution:
        # TODO: sort by quality, probabilistically (statistically) make a choice
        ...

    def memorize_move(self, move: Solution):
        self._memory.memorize(move)
        self.hall_of_fame.add(move)
        self.hall_of_fame.pop(0)

    def converged(self, move: Solution):
        return self.convergence_criterion.converged(new_result=move.quality)
