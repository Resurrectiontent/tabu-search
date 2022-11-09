from abc import ABC, abstractmethod


# TODO: implement SolutionQualityFactory


class BaseSolutionQualityMetric(ABC):
    """
    Maximized quality metric. I.e., `better_solution_quality_metric > worse_solution_quality_metric`
    """
    def __init__(self, name: str, value_str: str):
        self.name = name
        self._str = value_str

    @abstractmethod
    def __float__(self) -> float:
        """
        Float representation of the solution quality.
        Comparable with float representations of solution quality objects of same types.
        """
        ...

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
