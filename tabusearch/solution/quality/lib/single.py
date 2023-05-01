from functools import partial
from numbers import Number
from typing import Iterable, Callable

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from tabusearch.typing_ import TData
from tabusearch.solution.quality.single import SolutionQualityInfo


# TODO: rewrite for iterable
def sum_metric(name: str, weights: NDArray[Number] | Iterable[Number] | None = None, **kwargs) \
        -> Callable[[NDArray[Number]], SolutionQualityInfo]:
    weights = weights and (weights if isinstance(weights, ndarray) else np.array(weights))
    float_ = weights and (lambda x: (x * weights).sum()) or np.sum

    return partial(SolutionQualityInfo, name=name, float_=float_, **kwargs)


# TODO: fix typing hints
def custom_metric(name: str, evaluation: Callable[[TData], float], **kwargs) \
        -> Callable[[TData], SolutionQualityInfo]:
    single_factory = partial(SolutionQualityInfo, name=name, float_=evaluation, **kwargs)

    def iter_metric(x: list[TData]) -> list[float]:
        return [single_factory(s) for s in x]

    return iter_metric

