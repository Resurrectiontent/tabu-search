from abc import ABC, abstractmethod
from copy import copy
from operator import itemgetter
from typing import List, Set, Iterable, Callable

from mutation.base import Solution, TMoveId


class MemoryCriterion(ABC):
    _moves: List[Solution]
    _move_idx: Set[TMoveId]
    # TODO: Consider more elegant implementation
    _solution_id_getter: Callable[[Solution], TMoveId]

    # TODO: consider taking set of solutions to broaden implementation possibilities.
    @abstractmethod
    def _criterion(self, x_idx: Set[TMoveId]) -> Set[TMoveId]:
        """
        Returns a set of ids, which corresponds to this criterion type. The set can have intersection with x_idx
        :param x_idx:
        :return:
        """
        ...

    def memorize(self, move: Solution):
        self._moves.append(move)
        self._move_idx.update(self._solution_id_getter(move))

    def filter(self, x: Iterable[Solution]) -> Iterable[Solution]:
        good_idx, move_idx = self._get_all_and_good_move_idx(x)
        return [move for move, idx in zip(x, move_idx) if idx in good_idx]

    def _filter_list_idx(self, x: Iterable[Solution]) -> Set[int]:
        """
        Like filter, but uses solutions ids in ordered list, rather then solution ids via _solution_id_getter
        :param x:
        :return:
        """
        good_idx, move_idx = self._get_all_and_good_move_idx(x)
        return {i for i, idx in enumerate(move_idx) if idx in good_idx}

    def _get_all_and_good_move_idx(self, x: Iterable[Solution]):
        move_idx = [self._solution_id_getter(move) for move in x]
        good_idx = self._criterion(set(move_idx))
        return good_idx, move_idx

    def _negate_criterion(self):
        def negated_criterion(criterion):
            def wrapper(self, x: Set[TMoveId]) -> Set[TMoveId]:
                return x.difference(criterion(self, x))

            return wrapper

        self._criterion = negated_criterion(self._criterion)

    def __and__(self, other: 'MemoryCriterion') -> 'MemoryCriterion':
        return CumulativeMemoryCriterion(set.union, self, other)

    def __or__(self, other: 'MemoryCriterion') -> 'MemoryCriterion':
        return CumulativeMemoryCriterion(set.intersection, self, other)

    def __invert__(self) -> 'MemoryCriterion':
        result = copy(self)
        result._negate_criterion()
        return result


class CumulativeMemoryCriterion(MemoryCriterion):
    """
    Used to accumulate multiple memory criteria
    """
    def __init__(self, operation: Callable[[Set[TMoveId], Set[TMoveId]], Set[TMoveId]], *criteria):
        assert len(criteria) > 0

        self._operation = operation
        self._criteria: List[MemoryCriterion] = list(criteria)
        self._inverted = False

    def filter(self, x: Iterable[Solution]) -> Iterable[Solution]:
        good_list_idx = self._filter_list_idx(x)
        return itemgetter(*good_list_idx)(x)

    def memorize(self, move: Solution):
        for c in self._criteria:
            c.memorize(move)

    def _filter_list_idx(self, x: Iterable[Solution]) -> Set[int]:
        x = x if isinstance(x, list) else list(x)
        res = self._criteria[0]._filter_list_idx(x)

        for c in self._criteria[1:]:
            res = self._operation(res, c._filter_list_idx(x))

        return set(range(len(x))).difference(res) if self._inverted else res

    def _negate_criterion(self):
        self._inverted = not self._inverted

    def _criterion(self, x_idx: Set[TMoveId]) -> Set[TMoveId]:
        # Should never be called
        raise NotImplementedError('No implementation for cumulative CumulativeMemoryCriterion _criterion')
