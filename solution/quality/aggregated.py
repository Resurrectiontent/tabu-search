from typing import Callable, Iterable

from solution.quality.base import BaseSolutionQualityMetric


class AggregatedSolutionQuality(BaseSolutionQualityMetric):

    def __init__(self, name: str, *metrics, aggregation: [Callable[[Iterable[bool]], bool]]):
        value_str = '\n'.join(list(map(str, metrics)))
        super().__init__(name, value_str)

        self.metrics = list(metrics)
        self.aggregation = aggregation

        ...

    def _equals_to(self, other) -> bool:
        pass

    def _less_than(self, other) -> bool:
        pass

# TODO: Custom aggregation with aggregation:[Callable[[Iterable[bool]], bool]
# TODO: Shortcut for all, any, etc.
