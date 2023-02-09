from functools import partial
from typing import Union, Callable, Optional, TypeVar

from numpy import NAN

from solution.quality.base import BaseSolutionQualityInfo

TData = TypeVar('TData')


class SolutionQualityInfo(BaseSolutionQualityInfo):
    def __init__(self, data: TData,
                 name: str,
                 float_: Union[float, int, Callable[[TData], float]],
                 minimized: Optional[bool] = False,
                 value_str: Optional[Union[str, Callable[[TData], str]]] = str):
        value_str = value_str(data) if callable(value_str) else value_str
        super().__init__(f'{name}({"min" if minimized else "max"})', value_str)

        self._data = data
        self._float_f, self._float_n = (partial(float_, data), NAN) if callable(float_) else (None, float_)
        self._minimized = minimized

    @property
    def _float(self):
        if self._float_n is NAN:
            self._float_n = self._float_f()
        return self._float_n

    def _equals_to(self, other) -> bool:
        return float(self) == float(other)

    def _less_than(self, other) -> bool:
        return float(self) < float(other)

    def __float__(self) -> float:
        """
        Float representation of the solution quality.
        Comparable with float representations of solution quality objects of same types.
        """
        return -self._float if self._minimized else self._float
