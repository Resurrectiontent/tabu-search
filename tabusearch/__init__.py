from abc import ABC
from copy import copy
from itertools import chain
from operator import attrgetter
from sortedcontainers import SortedList
from typing import Iterable, Callable, Generic

from tabusearch.convergence import IterativeConvergence
from tabusearch.convergence.base import ConvergenceCriterion
from tabusearch.memory import AspirationCriterion, TabuList
from tabusearch.memory.base import BaseMemoryCriterion
from tabusearch.mutation.base import MutationBehaviour
from tabusearch.solution.base import Solution
from tabusearch.solution.factory import SolutionFactory
from tabusearch.solution.quality.aggregated import BaseAggregatedSolutionQualityInfo
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.solution.quality.single import SolutionQualityInfo
from tabusearch.solution.selection import SolutionSelection
from tabusearch.typing_ import TData


# TODO: consider implementing epsilon-greedy strategy
# TODO: implement lazy quality calculation with corresponding parameter in factory constructor
# TODO: implement dimensionality reduction


# TODO: improve optimization history
# FIXME: actually maximizes function
# FIXME: sometimes finds no neighbours
class TabuSearch(ABC, Generic[TData]):
    hall_of_fame_size: int

    hall_of_fame: SortedList[Solution[TData]]  # sorted by ascending quality
    convergence_criterion: ConvergenceCriterion
    solution_factory: SolutionFactory[TData]
    mutation_behaviour: Iterable[MutationBehaviour[TData]]
    aspiration: AspirationCriterion
    tabu: TabuList
    solution_selection: SolutionSelection
    _memory: BaseMemoryCriterion

    _history: list

    # TODO: consider further ctor conveniences
    def __init__(self, mutation_behaviour: Callable[[SolutionFactory], MutationBehaviour]
                                           | Iterable[Callable[[SolutionFactory], MutationBehaviour]],
                 metric: Callable[[TData], SolutionQualityInfo]
                         | Iterable[Callable[[TData], SolutionQualityInfo]],
                 hall_of_fame_size: int = 5,
                 convergence_criterion: ConvergenceCriterion | int = 100,
                 tabu_time: Callable[[Solution[TData]], int] | int = 5,
                 selection: Callable[[], int] | None = None,
                 metric_aggregation: Callable[[Iterable[BaseSolutionQualityInfo]], BaseAggregatedSolutionQualityInfo]
                                     | None = None):
        assert not isinstance(metric, Iterable) or metric_aggregation, \
            'Should provide metrics_aggregation, if passing several items in metric arg.'

        self.hall_of_fame_size = hall_of_fame_size

        self.hall_of_fame = SortedList(key=attrgetter('quality'))
        # TODO: move convergence to arguments
        self.convergence_criterion = IterativeConvergence(convergence_criterion) \
            if isinstance(convergence_criterion, int) \
            else convergence_criterion
        self.solution_factory = SolutionFactory((*metric,) if isinstance(metric, Iterable) else metric,
                                                metrics_aggregation=metric_aggregation)
        mutation_behaviour = mutation_behaviour if isinstance(mutation_behaviour, Iterable) else [mutation_behaviour]
        self.mutation_behaviour = [mutation(self.solution_factory) for mutation in mutation_behaviour]
        self.aspiration = AspirationCriterion()
        self.tabu = TabuList(tabu_time)
        self.solution_selection = SolutionSelection(selection or (lambda _: 0))

    @property
    def resulting_memory_criterion(self):
        if not hasattr(self, '_memory') or self._memory is None:
            self._memory = self.tabu.unite(self.aspiration)
        return self._memory

    def optimize(self, x0: TData) -> Solution[TData]:
        # TODO: move to memorize_move
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
        return self.hall_of_fame[-1]

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
