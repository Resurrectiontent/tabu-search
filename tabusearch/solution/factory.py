from itertools import chain
from typing import Callable, Iterable, Generic

from tabusearch.solution.base import Solution
from tabusearch.solution.id import SolutionId
from tabusearch.solution.quality.aggregated import BaseAggregatedSolutionQualityInfo
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.solution.quality.factory import SolutionQualityFactory
from tabusearch.typing_ import TData


class SolutionFactory(Generic[TData]):
    quality_factory: SolutionQualityFactory

    # TODO: abstract *metrics creation to utility functions
    #  consider partial for BaseSolutionQualityInfo-inherited ctors
    def __init__(self, *metrics: Callable[[list[TData]], list[BaseSolutionQualityInfo]],
                 metrics_aggregation: Callable[[Iterable[Iterable[BaseSolutionQualityInfo]]],
                                               Iterable[BaseAggregatedSolutionQualityInfo]]
                                      | None = None):
        self.quality_factory = SolutionQualityFactory(*metrics,
                                                      metrics_aggregation=metrics_aggregation)

    def __call__(self, generated: list[tuple[str, list[tuple[TData, str]]]]) -> list[Solution[TData]]:
        solutions: list[tuple[SolutionId, TData]] = list(chain(*[[(SolutionId(generator_name, *name_suffix), position)
                                                                  for position, *name_suffix in solutions]
                                                                 for generator_name, solutions in generated]))

        qualities = self.quality_factory([position for _, position in solutions])
        return [Solution(solution_id, position, quality)
                for (solution_id, position), quality in zip(solutions, qualities)]

    def initial(self, position: TData) -> Solution[TData]:
        return Solution(SolutionId('Init'), position, self.quality_factory.single(position))
