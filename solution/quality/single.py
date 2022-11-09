from typing import Union, Callable, Optional, TypeVar

from solution.quality.base import BaseSolutionQualityMetric

TData = TypeVar('TData')

# TODO: implement LazySolutionQualityMetric


class SolutionQualityMetric(BaseSolutionQualityMetric):
    def __init__(self, name: str,
                 data: TData,
                 float_: Union[float, int, Callable[[TData], float]],
                 minimized: Optional[bool] = False,
                 value_str: Optional[Union[str, Callable[[TData], float]]] = str):
        value_str = value_str(data) if callable(value_str) else value_str
        super().__init__(name, value_str)

        self._data = data
        self._float = float_(data) if callable(float_) else float_
        self._minimized = minimized

    def __float__(self) -> float:
        """
        Float representation of the solution quality.
        Comparable with float representations of solution quality objects of same types.
        """
        return -self._float if self._minimized else self._float
