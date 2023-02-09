# TODO: add simple non-aggregated metrics' shortcuts:
#  sum/weighted sum
#  ? custom function ?
from functools import partial
from numbers import Number
from typing import Iterable, Callable

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from solution.quality.single import SolutionQualityInfo


def sum_metric(name: str, weights: NDArray[Number] | Iterable[Number] | None = None, **kwargs) \
        -> Callable[[NDArray[Number]], SolutionQualityInfo]:
    weights = weights and (weights if isinstance(weights, ndarray) else np.array(weights))
    float_ = weights and (lambda x: (x * weights).sum()) or np.sum

    return partial(SolutionQualityInfo, name=name, float_=float_, **kwargs)


def custom_metric(name: str, evaluation: Callable[[NDArray[Number]], float], **kwargs) \
        -> Callable[[NDArray[Number]], SolutionQualityInfo]:
    return partial(SolutionQualityInfo, name=name, float_=evaluation, **kwargs)
