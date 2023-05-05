from _operator import attrgetter, itemgetter
from abc import ABC, abstractmethod
from copy import copy
from functools import wraps, reduce
from typing import Iterable, Set, Tuple, List, Callable

from memory.base import BaseMemoryCriterion
from solution.id import SolutionId
from tabusearch import Solution


class BaseFilteringMemoryCriterion(BaseMemoryCriterion, ABC):
    _inverted: bool

    def __init__(self):
        self._inverted = False

    @abstractmethod
    def filter(self, x: Iterable[Solution]) -> Iterable[Solution]:
        """
        Filters solutions due to conditions of the MemoryCriterion.
        :param x: Proposed set (collection) of solutions.
        :return: Subset of allowed solutions.
        """

    @abstractmethod
    def _filter_list_idx(self, x: Iterable[Solution]) -> Set[int]:
        """
        Like filter, but uses solutions ids in ordered list, rather than solution ids via _solution_id_getter
        :param x:
        :return:
        """
        ...

    @abstractmethod
    def _negate_criterion(self):
        """
        Negates criterion, i.e. subtracts its former return value from the input set of solutions.
        """
        ...

    def unite(self, other: 'BaseFilteringMemoryCriterion') -> 'BaseFilteringMemoryCriterion':
        """
        Represents union of criteria.
        :param other: other criterion
        :return: CumulativeMemoryCriterion, which `.filter()` returns union of sets, filtered by the criteria.
        """
        return CumulativeFilteringMemoryCriterion(set.union, self, other)

    def intersect(self, other: 'BaseFilteringMemoryCriterion') -> 'BaseFilteringMemoryCriterion':
        """
        Represents intersection of criteria.
        :param other: other criterion
        :return: CumulativeMemoryCriterion, which `.filter()` returns intersection of sets, filtered by the criteria.
        """
        return CumulativeFilteringMemoryCriterion(set.intersection, self, other)

    def inverted(self) -> 'BaseFilteringMemoryCriterion':
        """
        Represents criterion negation.
        :return: Copy of the criterion with inverted filtering.
        """
        result = copy(self)
        result._negate_criterion()
        return result


class FilteringMemoryCriterion(BaseFilteringMemoryCriterion, ABC):
    @abstractmethod
    def _criterion(self, x: Iterable[Solution]) -> Set[SolutionId]:
        """
        Returns a set of ids, which allowed by this criterion type. The set can have intersection with ids of x
        :param x:
        :return:
        """
        ...

    def filter(self, x: Iterable[Solution]) -> Iterable[Solution]:
        good_idx, move_idx = self._get_all_and_good_move_idx(x)
        return [move for move, idx in zip(x, move_idx) if idx in good_idx]

    def _filter_list_idx(self, x: Iterable[Solution]) -> Set[int]:
        good_idx, move_idx = self._get_all_and_good_move_idx(x)
        return {i for i, idx in enumerate(move_idx) if idx in good_idx}

    def _negate_criterion(self):
        def negated_criterion(criterion):
            @wraps(criterion)
            def wrapper(_self: FilteringMemoryCriterion, x: Iterable[Solution]) -> Set[SolutionId]:
                return {el.id for el in x}.difference(criterion(_self, x))

            return wrapper

        # noinspection PyUnresolvedReferences
        self._criterion = self._criterion.__wrapped__ \
            if self._inverted else \
            negated_criterion(self._criterion)
        self._inverted = not self._inverted

    def _get_all_and_good_move_idx(self, x: Iterable[Solution]) -> Tuple[Set[SolutionId], List[SolutionId]]:
        move_idx = list(map(attrgetter('id'), x))
        # noinspection PyArgumentList
        good_idx = self._criterion(x)
        return good_idx, move_idx


class CumulativeFilteringMemoryCriterion(BaseFilteringMemoryCriterion):
    """
    Used to accumulate multiple memory criteria
    """

    def __init__(self, operation: Callable[[set, set], set], *criteria):
        assert len(criteria) > 0
        super().__init__()

        self._operation = operation
        self._criteria: List[BaseFilteringMemoryCriterion] = list(criteria)

    def filter(self, x: Iterable[Solution]) -> Iterable[Solution]:
        x = x if isinstance(x, list) else list(x)
        good_list_idx = self._filter_list_idx(x)
        num = len(good_list_idx)
        return itemgetter(*good_list_idx)(x) if num > 1 else \
            ((x[good_list_idx.pop()],) if num else [])

    def memorize(self, move: Solution):
        for c in self._criteria:
            c.memorize(move)

    def _filter_list_idx(self, x: Iterable[Solution]) -> Set[int]:
        x = x if isinstance(x, list) else list(x)

        res = reduce(self._operation, [c._filter_list_idx(x) for c in self._criteria])
        return set(range(len(x))).difference(res) if self._inverted else res

    def _negate_criterion(self):
        self._inverted = not self._inverted
