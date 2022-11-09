from typing import Callable, TypeVar, Union, Optional


# TODO: implement SolutionQualityFactory
# TODO: implement AggregateSolutionQuality and different aggregation types

TData = TypeVar('TData')


class SolutionQuality:
    def __init__(self, name: str,
                 data: TData,
                 float_: Union[float, int, Callable[[TData], float]],
                 maximize: Optional[bool] = False,
                 str_: Optional[Union[str, Callable[[TData], float]]] = str):
        self.name = name
        self._data = data
        self._float = float_(data) if callable(float_) else float_
        self._maximize = maximize
        self._str = str_(data) if callable(str_) else str_

    def __float__(self) -> float:
        """
        Float representation of the solution quality.
        Comparable with float representations of solution quality objects of same types.
        """
        return self._float

    def __str__(self) -> str:
        """
        String exploration of the solution quality.
        """
        return f'{self.name}: {self._str}'

    def _type_check(self, other):
        """
        Check whether solution qualities have same types.
        :param other: Operand.
        :return: Throws `AssertionException` if types do not match.
        """
        self_type = type(self)
        assert isinstance(other, self_type), 'Comparison of solution qualities is possible ony for' \
                                             ' SolutionQuality descendants of same types' \
                                             f' (were {self_type.__name__} and {type(other).__name__}).'

    def __eq__(self, other) -> bool:
        """
        Compares solution qualities of same types.
        :param other: Comparable.
        :return: Whether this solution quality equals to `other' solution.
        """
        self._type_check(other)
        return float(self) == float(other)

    def __lt__(self, other) -> bool:
        """
        Compares solution qualities of same types.
        :param other: Comparable.
        :return: Whether this solution quality is worse than `other' solution.
        """
        self._type_check(other)
        return float(self) < float(other)
