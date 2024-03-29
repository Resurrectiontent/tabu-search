from functools import partial
from typing import Union, Callable, Optional, Generic

from numpy import NAN

from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.typing_ import TData


class SolutionQualityInfo(BaseSolutionQualityInfo, Generic[TData]):
    def __init__(self, data: TData,
                 name: str,
                 float_: float | int | Callable[[TData], float],
                 minimized: bool | None = False,
                 value_str: str | Callable[[TData], str] | None = str):
        value_str = value_str(data) if callable(value_str) else value_str
        super().__init__(f'{name}({"min" if minimized else "max"})', value_str)

        self._data = data
        self._float_f, self._float_n = (partial(float_, data), NAN) if callable(float_) else (None, float_)
        self._minimized = minimized

    def quality_like(self, **kwargs):
        return SolutionQualityInfo(**dict(dict(data=self._data,
                                               name=self.name,
                                               float_=self._float,
                                               minimized=self._minimized,
                                               value_str=self._str), **kwargs))

    @property
    def value(self):
        """
        The real value of quality function, not depending on whether it is minimized or not.
        """
        return self._float

    @property
    def _float(self):
        if self._float_n is NAN:
            self._float_n = self._float_f()
        return float(self._float_n)

    def _equals_to(self, other) -> bool:
        return float(self) == float(other)

    def _less_than(self, other) -> bool:
        return float(self) < float(other)

    def __float__(self) -> float:
        """
        Float representation of the solution quality.
        Is negated, if the function is minimised to support proper comparison.
        Comparable with float representations of solution quality objects of same types.
        """
        return -self._float if self._minimized else self._float

    def __str__(self) -> str:
        return f'{self.name} ({self.value})'
