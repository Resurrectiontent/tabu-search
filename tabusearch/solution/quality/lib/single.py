from functools import partial
from multiprocessing.pool import ThreadPool
from numbers import Number
from typing import Iterable, Callable

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from tabusearch.typing_ import TData
from tabusearch.solution.quality.single import SolutionQualityInfo


# TODO: rewrite for iterable
def sum_metric(name: str, weights: NDArray[Number] | Iterable[Number] | None = None, **kwargs) \
        -> Callable[[list[NDArray[Number]]], list[SolutionQualityInfo]]:
    weights = weights and (weights if isinstance(weights, ndarray) else np.array(weights))
    float_ = weights and (lambda x: (x * weights).sum()) or np.sum
    single_factory = partial(SolutionQualityInfo, name=name, float_=float_, **kwargs)

    def iter_metric(x: list[TData]) -> list[SolutionQualityInfo]:
        return list(map(single_factory, x))

    return iter_metric


def custom_metric(name: str, evaluation: Callable[[TData], float], **kwargs) \
        -> Callable[[list[TData]], list[SolutionQualityInfo]]:
    single_factory = partial(SolutionQualityInfo, name=name, float_=evaluation, **kwargs)

    def iter_metric(x: list[TData]) -> list[SolutionQualityInfo]:
        return list(map(single_factory, x))

    return iter_metric


def custom_metric_parallel(name: str, evaluation: Callable[[TData], float], **kwargs) \
        -> Callable[[list[TData]], list[SolutionQualityInfo]]:
    single_factory = partial(SolutionQualityInfo, name=name, float_=evaluation, **kwargs)
    pool = ThreadPool(8)

    def iter_metric(x: list[TData]) -> list[SolutionQualityInfo]:
        return list(pool.map(single_factory, x))

    return iter_metric

