from abc import ABC, abstractmethod
from operator import attrgetter
from typing import TypeVar, Generic, Iterable

from tabusearch.solution.base import Solution
from tabusearch.memory.base import BaseMemoryCriterion
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.typing_ import TData


TEvaluatedData = TypeVar('TEvaluatedData')


class BaseEvaluatingMemoryCriterion(BaseMemoryCriterion, Generic[TData, TEvaluatedData], ABC):
    def __call__(self, x: list[Solution[TData]]) -> list[BaseSolutionQualityInfo]:
        """
        Evaluates list of solutions.
        :param x:
        :return:
        """
        return self.evaluate(zip(x, self.convert_to_evaluated_data_type(map(attrgetter('position'), x))))

    def convert_to_evaluated_data_type(self, x: Iterable[TData]) -> Iterable[TEvaluatedData]:
        """
        Converts the optimised data type to the evaluated one, if they are different.
        Default case for TData == TEvaluatedData.
        :param x: The optimised data.
        :return: The evaluated data.
        """
        return x

    @abstractmethod
    def evaluate(self, x: Iterable[tuple[Solution[TData], TEvaluatedData]]) -> list[BaseSolutionQualityInfo]:
        ...
