from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterable

from tabusearch.memory.base import BaseMemoryCriterion
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.typing_ import TData


# Only the data core (e.g., ndarray)
TEvaluatedData = TypeVar('TEvaluatedData')
# ndarray or tuple[ndarray, BaseSolutionQualityInfo]
TFullEvaluatedInfo = TypeVar('TFullEvaluatedInfo')
# TData or tuple[TData, BaseSolutionQualityInfo]
TReceivedData = TypeVar('TReceivedData')


class BaseEvaluatingMemoryCriterion(BaseMemoryCriterion,
                                    Generic[TReceivedData, TFullEvaluatedInfo, TEvaluatedData],
                                    ABC):
    @abstractmethod
    def __call__(self, x: list[TReceivedData]) -> list[BaseSolutionQualityInfo]:
        """
        Evaluates list of solutions.
        :param x:
        :return:
        """
        ...

    @abstractmethod
    def evaluate(self, x: Iterable[TFullEvaluatedInfo]) -> list[BaseSolutionQualityInfo]:
        ...


class FirstOrderEvaluatingMemoryCriterion(BaseEvaluatingMemoryCriterion[TData, TEvaluatedData, TEvaluatedData],
                                          Generic[TData, TEvaluatedData],
                                          ABC):
    def __call__(self, x: list[TData]) -> list[BaseSolutionQualityInfo]:
        """
        Evaluates list of solutions.
        :param x:
        :return:
        """
        return self.evaluate(self.convert_to_evaluated_data_type(x))

    def convert_to_evaluated_data_type(self, x: Iterable[TData]) -> Iterable[TEvaluatedData]:
        """
        Converts the optimised data type to the evaluated one, if they are different.
        Default case for TData == TEvaluatedData.
        :param x: The optimised data.
        :return: The evaluated data.
        """
        return x


class SecondOrderEvaluatingMemoryCriterion(BaseEvaluatingMemoryCriterion[tuple[TData, BaseSolutionQualityInfo],
                                                                         tuple[TEvaluatedData, BaseSolutionQualityInfo],
                                                                         TEvaluatedData],
                                           Generic[TData, TEvaluatedData],
                                           ABC):
    def __call__(self, x: list[tuple[TData, BaseSolutionQualityInfo]]) -> list[BaseSolutionQualityInfo]:
        """
        Evaluates list of solutions.
        :param x:
        :return:
        """
        data, quality = zip(*x)
        return self.evaluate(zip(self.convert_to_evaluated_data_type(data), quality))

    def convert_to_evaluated_data_type(self, x: Iterable[TData]) -> Iterable[TEvaluatedData]:
        """
        Converts the optimised data type to the evaluated one, if they are different.
        Default case for TData == TEvaluatedData.
        :param x: The optimised data.
        :return: The evaluated data.
        """
        return x
