from tabusearch.solution.quality.aggregated import BaseAggregatedSolutionQualityInfo
from tabusearch.solution.quality.single import SolutionQualityInfo


class ComplexSolutionQualityInfo(SolutionQualityInfo):
    """
    Represents a complex solution quality.
    It contains 'main' SolutionQualityInfo, which is used as a display quality metric
      and `full` one, which accounts several metrics to adjust solution ordering by quality.
    """
    main: SolutionQualityInfo
    full: BaseAggregatedSolutionQualityInfo

    # noinspection PyMissingConstructor
    def __init__(self, main: SolutionQualityInfo, full: BaseAggregatedSolutionQualityInfo):
        self.main = main
        self.full = full

        self.name = f'Complex [{main.name} ({full.name})]'
        self._str = full._str

    @property
    def value(self):
        """
        The real value of main quality function, not depending on whether it is minimized or not.
        """
        return self.main.value

    @property
    def _float(self):
        return self.main._float

    def _equals_to(self, other: 'ComplexSolutionQualityInfo') -> bool:
        return self.full == other.full

    def _less_than(self, other: 'ComplexSolutionQualityInfo') -> bool:
        return self.full < other.full
