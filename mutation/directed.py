from abc import ABC
from enum import IntEnum
from functools import partialmethod, wraps
from typing import Callable, Tuple, List, Iterable

from numpy import ndarray

from mutation.base import MutationBehaviour


def return_self_method(func):
    assert callable(func), f'Argument func should be callable. Type {type(func).__name__} is not callable.'

    @wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        return func.__self__
    return wrapper


class MutationDirection(IntEnum):
    Negative = 1
    Bidirectional = 2
    Positive = 3

    @property
    def is_positive(self):
        return self.value > 1

    @property
    def is_negative(self):
        return self.value < 3

    @property
    def is_single_directional(self):
        return self.value != 2

    def reverse(self):
        if self.value == 1:
            return MutationDirection.Positive
        if self.value == 3:
            return MutationDirection.Negative
        return self


class BidirectionalMutationBehaviour(MutationBehaviour, ABC):
    """
    Usage:
      Override one of methods _generate_one_direction_mutation or _generate_one_direction_mutations
      in dependence on whether mutation generates single new solution or multiple new solutions.
      By default, one of them (which exists) is called in _generate_mutations.
      To change this, override _generate_mutations
    """
    _generate_one_direction_mutation: Callable[[ndarray, bool], Tuple[str, ndarray]]
    _generate_one_direction_mutations: Callable[[ndarray, bool], List[Tuple[str, ndarray]]]

    _direction: MutationDirection = MutationDirection.Positive

    def __init__(self, quality: Callable[[ndarray], float], mutation_direction: MutationDirection = None):
        """
        Initializes BidirectionalMutationBehaviour.
        :param quality: Quality function for evaluating proposed solutions.
        :param mutation_direction: Direction of mutation. MutationDirection.Positive by default
        """
        super().__init__(quality)
        if mutation_direction:
            self._direction = mutation_direction

        def self_decorate(name):
            setattr(self, name, return_self_method(getattr(self, name)))

        map(self_decorate, ['to_negative_direction', 'to_positive_direction', 'to_bidirectional', 'reverse_direction'])

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        self._direction = value

    to_negative_direction = partialmethod(direction.fset, MutationDirection.Negative)
    to_positive_direction = partialmethod(direction.fset, MutationDirection.Positive)
    to_bidirectional = partialmethod(direction.fset, MutationDirection.Bidirectional)
    reverse_direction = partialmethod(_direction.reverse)

    def _generate_mutations(self, x: ndarray) -> Iterable[Tuple[str, ndarray]]:
        """
        Default implementation for bidirectional mutations. Override, if you need to change it.
        """
        assert self._generate_one_direction_mutation or self._generate_one_direction_mutations, \
            'Should implement either _generate_one_direction_mutation or _generate_one_direction_mutations methods' \
            ' to use default logics or override _generate_mutations to implement custom logics.'
        r = []

        recorder, generator = (r.extend, self._generate_one_direction_mutations) \
            if self._generate_one_direction_mutations \
            else (r.append, self._generate_one_direction_mutation)

        # Executes both, if bidirectional
        if self._direction.is_negative:
            recorder(generator(x, True))
        if self._direction.is_positive:
            recorder(generator(x, False))

        return r
