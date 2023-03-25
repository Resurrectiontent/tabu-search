from numpy.random import default_rng, Generator
from typing import Callable

from sampo.scheduler.genetic.converter import ChromosomeType
from sampo.scheduler.genetic.operators import copy_chromosome


def variable_partitioning_order_neighbourhood(ind: ChromosomeType,
                                              is_order_correct: Callable[[ChromosomeType], bool],
                                              max_distance: int = 10,
                                              one_distance_trials: int = 4,
                                              max_one_distance_offset: int | None = 5,
                                              rng: Generator = default_rng()):
    size = ind[0].size
    assert max_distance >= 2
    assert size >= max_distance + one_distance_trials
    max_one_distance_offset = max_one_distance_offset or max_distance

    result = []

    for distance in range(2, max_distance + 1):
        trials = rng.choice(size - distance + 1, one_distance_trials, replace=False)
        for trial_start in trials:
            for offset in range(1, min(distance, max_one_distance_offset) + 1):
                shifted_right = copy_chromosome(ind)
                shifted_left = copy_chromosome(ind)

                shifted_right[0][trial_start:trial_start + distance] \
                    = shifted_right[0][shift(list(range(trial_start, trial_start + distance)), offset, False)]
                shifted_left[0][trial_start:trial_start + distance] \
                    = shifted_left[0][shift(list(range(trial_start, trial_start + distance)), offset, True)]

                if is_order_correct(shifted_right):
                    result.append((shifted_right, f'+{trial_start}:{distance},{offset}'))
                if is_order_correct(shifted_left):
                    result.append((shifted_left, f'-{trial_start}:{distance},{offset}'))

    return result


def shift(lst: list, offset: int, direction_left: bool):
    return lst[offset:] + lst[:offset] \
        if direction_left \
        else lst[-offset:] + lst[:-offset]
