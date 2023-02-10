from abc import ABC
from copy import copy
from itertools import chain
from operator import attrgetter
from typing import Iterable, Callable
from numbers import Number

from numpy import ndarray
from numpy.typing import NDArray
from sortedcontainers import SortedList

from convergence.base import ConvergenceCriterion
from convergence.single import IterativeConvergence
from memory.aspiration import AspirationCriterion
from memory.base import BaseMemoryCriterion
from memory.tabu import TabuList
from mutation.base import MutationBehaviour
from mutation.neighbourhood import NearestNeighboursMutation
from solution.base import Solution
from solution.factory import SolutionFactory
from solution.quality.lib.single import sum_metric, custom_metric
from solution.selection import SolutionSelection


# TODO: consider implementing epsilon-greedy strategy
# TODO: implement lazy quality calculation with corresponding parameter in factory constructor
# TODO: implement dimensionality reduction


# TODO: improve optimization history
# FIXME: actually maximizes function
# FIXME: sometimes finds no neighbours
class TabuSearch(ABC):
    hall_of_fame_size: int

    hall_of_fame: SortedList[Solution]  # sorted by ascending quality
    convergence_criterion: ConvergenceCriterion
    solution_factory: SolutionFactory
    mutation_behaviour: Iterable[MutationBehaviour]
    aspiration: AspirationCriterion
    tabu: TabuList
    solution_selection: SolutionSelection
    _memory: BaseMemoryCriterion

    _history: list

    # TODO: implement constructor
    def __init__(self, hall_of_fame_size: int = 5,
                 max_iter: int = 100,
                 tabu_time: int = 5,
                 quality: Callable[[NDArray[Number]], float] | None = None):
        self.hall_of_fame_size = hall_of_fame_size

        self.hall_of_fame = SortedList(key=attrgetter('quality'))
        self.convergence_criterion = IterativeConvergence(max_iter)
        self.solution_factory = SolutionFactory(quality and custom_metric('Custom metric', quality)
                                                        or sum_metric('Sum dimensions'))
        self.mutation_behaviour = [NearestNeighboursMutation(self.solution_factory)]
        self.aspiration = AspirationCriterion()
        self.tabu = TabuList(tabu_time)
        self.solution_selection = SolutionSelection(lambda _: 0)

    @property
    def resulting_memory_criterion(self):
        if not hasattr(self, '_memory') or self._memory is None:
            self._memory = self.tabu.unite(self.aspiration)
        return self._memory

    def optimize(self, x0: ndarray):
        history = []

        x = self.solution_factory.initial(x0)
        history.append(copy(x))

        while True:

            neighbours = self.get_neighbours(x)
            x = self.choose(neighbours)
            self.memorize_move(x)

            history.append(copy(x))

            if self.converged(x):
                break

        self._history = history
        return x

    def get_neighbours(self, x: Solution) -> Iterable[Solution]:
        possible_moves = chain(*[behaviour.mutate(x) for behaviour in self.mutation_behaviour])
        return self.resulting_memory_criterion.filter(possible_moves)

    def choose(self, neighbours: Iterable[Solution]) -> Solution:
        neighbours = SortedList(neighbours, key=attrgetter('quality'))
        return self.solution_selection(neighbours)

    def memorize_move(self, move: Solution):
        self._memory.memorize(move)
        self.hall_of_fame.add(move)

        if len(self.hall_of_fame) > self.hall_of_fame_size:
            self.hall_of_fame.pop(0)

    def converged(self, move: Solution):
        return self.convergence_criterion.converged(new_result=move.quality)
