from abc import ABC, abstractmethod
from copy import copy
from operator import itemgetter
from typing import List, Set, Hashable, Iterable, Callable, TypeVar

from moves.base import Move
from utility.set import unite, intersect

TMoveId = TypeVar('TMoveId', bound=Hashable)


class MemoryCriterion(ABC):
    _moves: List[Move]
    _move_idx: Set[TMoveId]
    _move_id_getter: Callable[[Move], TMoveId]

    @abstractmethod
    def _criterion(self, x_idx: Set[TMoveId]) -> Set[TMoveId]:
        ...

    def memorize(self, move: Move):
        self._moves.append(move)
        self._move_idx.update(self._move_id_getter(move))

    def filter(self, x: Iterable[Move]) -> Iterable[Move]:
        good_idx, move_idx = self._get_all_and_good_move_idx(x)
        return [move for move, idx in zip(x, move_idx) if idx in good_idx]

    def _filter_list_idx(self, x: Iterable[Move]) -> Set[int]:
        good_idx, move_idx = self._get_all_and_good_move_idx(x)
        return {i for i, idx in enumerate(move_idx) if idx in good_idx}

    def _get_all_and_good_move_idx(self, x: Iterable[Move]):
        move_idx = [self._move_id_getter(move) for move in x]
        good_idx = self._criterion(set(move_idx))
        return good_idx, move_idx

    def _negate_criterion(self):
        def negated_criterion(criterion):
            def wrapper(self, x: Set[TMoveId]) -> Set[TMoveId]:
                return x.difference(criterion(self, x))

            return wrapper

        self._criterion = negated_criterion(self._criterion)

    def __and__(self, other: 'MemoryCriterion') -> 'MemoryCriterion':
        return CumulativeMemoryCriterion(unite, self, other)

    def __or__(self, other: 'MemoryCriterion') -> 'MemoryCriterion':
        return CumulativeMemoryCriterion(intersect, self, other)

    def __invert__(self) -> 'MemoryCriterion':
        result = copy(self)
        result._negate_criterion()
        return result


class CumulativeMemoryCriterion(MemoryCriterion):
    def __init__(self, operation: Callable[[Set[TMoveId], Set[TMoveId]], Set[TMoveId]], *criteria):
        assert len(criteria) > 0

        self._operation = operation
        self._criteria: List[MemoryCriterion] = list(criteria)
        self._inverted = False

    def filter(self, x: Iterable[Move]) -> Iterable[Move]:
        good_list_idx = self._filter_list_idx(x)
        return itemgetter(*good_list_idx)(x)

    def memorize(self, move: Move):
        for c in self._criteria:
            c.memorize(move)

    def _filter_list_idx(self, x: Iterable[Move]) -> Set[int]:
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
