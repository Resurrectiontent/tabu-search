from abc import ABC, abstractmethod
from copy import copy
from functools import wraps, reduce
from operator import itemgetter
from typing import List, Set, Iterable, Callable, Generic, Tuple

from mutation.base import TMoveId
from solution.base import Solution


class BaseMemoryCriterion(ABC, Generic[TMoveId]):
    _inverted: bool

    def __init__(self):
        self._inverted = False

    @abstractmethod
    def memorize(self, move: Solution):
        """
        Memories selected move to account it in memory criterion logics. Also, makes a "tick".
        :param move: The selected solution to memorize.
        """

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

    def __and__(self, other: 'BaseMemoryCriterion') -> 'BaseMemoryCriterion':
        """
        Represents union of criteria.
        :param other: other criterion
        :return: CumulativeMemoryCriterion, which `.filter()` returns union of sets, filtered by the criteria.
        """
        return CumulativeMemoryCriterion(set.union, self, other)

    def __or__(self, other: 'BaseMemoryCriterion') -> 'BaseMemoryCriterion':
        """
        Represents intersection of criteria.
        :param other: other criterion
        :return: CumulativeMemoryCriterion, which `.filter()` returns intersection of sets, filtered by the criteria.
        """
        return CumulativeMemoryCriterion(set.intersection, self, other)

    def __invert__(self) -> 'BaseMemoryCriterion':
        """
        Represents criterion negation.
        :return: Copy of the criterion with inverted filtering.
        """
        result = copy(self)
        result._negate_criterion()
        return result


class MemoryCriterion(BaseMemoryCriterion[TMoveId], ABC):
    # TODO: Consider more elegant implementation
    _solution_id_getter: Callable[[Solution], TMoveId]
    # TODO: Consider dropping, if not needed + consider moving logics from `_memorize` to `memorize`
    _solution_history: List[Solution]
    _solution_idx_history: Set[TMoveId]

    def __init__(self, solution_id_getter: Callable[[Solution], TMoveId]):
        super().__init__()

        self._solution_id_getter = solution_id_getter
        self._solution_idx_history = set()
        self._solution_history = []

    @abstractmethod
    def _criterion(self, x: Iterable[Solution]) -> Set[TMoveId]:
        """
        Returns a set of ids, which allowed by this criterion type. The set can have intersection with ids of x
        :param x:
        :return:
        """
        ...

    @abstractmethod
    def _memorize(self, move: Solution):
        """
        Implementation-specific part of move memorization.
        :param move: The selected solution.
        """
        ...

    def memorize(self, move: Solution):
        self._solution_history.append(move)
        self._solution_idx_history.update(self._solution_id_getter(move))
        self._memorize(move)

    def filter(self, x: Iterable[Solution]) -> Iterable[Solution]:
        good_idx, move_idx = self._get_all_and_good_move_idx(x)
        return [move for move, idx in zip(x, move_idx) if idx in good_idx]

    def _filter_list_idx(self, x: Iterable[Solution]) -> Set[int]:
        good_idx, move_idx = self._get_all_and_good_move_idx(x)
        return {i for i, idx in enumerate(move_idx) if idx in good_idx}

    def _negate_criterion(self):
        def negated_criterion(criterion):
            @wraps(criterion)
            def wrapper(_self: MemoryCriterion, x: Iterable[Solution]) -> Set[TMoveId]:
                return {_self._solution_id_getter(el) for el in x}.difference(criterion(_self, x))

            return wrapper

        # noinspection PyUnresolvedReferences
        self._criterion = self._criterion.__wrapped__ \
            if self._inverted else \
            negated_criterion(self._criterion)
        self._inverted = not self._inverted

    def _get_all_and_good_move_idx(self, x: Iterable[Solution]) -> Tuple[Set[TMoveId], List[TMoveId]]:
        move_idx = [self._solution_id_getter(move) for move in x]
        # noinspection PyArgumentList
        good_idx = self._criterion(set(move_idx))
        return good_idx, move_idx


class CumulativeMemoryCriterion(BaseMemoryCriterion[TMoveId]):
    """
    Used to accumulate multiple memory criteria
    """
    def __init__(self, operation: Callable[[Set[TMoveId], Set[TMoveId]], Set[TMoveId]], *criteria):
        assert len(criteria) > 0
        super().__init__()

        self._operation = operation
        self._criteria: List[BaseMemoryCriterion] = list(criteria)

    def filter(self, x: Iterable[Solution]) -> Iterable[Solution]:
        good_list_idx = self._filter_list_idx(x)
        return itemgetter(*good_list_idx)(x)

    def memorize(self, move: Solution):
        for c in self._criteria:
            c.memorize(move)

    def _filter_list_idx(self, x: Iterable[Solution]) -> Set[int]:
        x = x if isinstance(x, list) else list(x)

        res = reduce(self._operation, [c._filter_list_idx(x) for c in self._criteria])
        return set(range(len(x))).difference(res) if self._inverted else res

    def _negate_criterion(self):
        self._inverted = not self._inverted
