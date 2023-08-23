from functools import wraps

import numpy as np
from numbers import Number

from numpy.typing import NDArray

# TODO: consider introducing real constrain checks before decorated function execution


def bounds(min_bounds: NDArray[Number] | None = None, max_bounds: NDArray[Number] | None = None):
    if min_bounds is not None and max_bounds is not None:
        assert all(min_bounds <= max_bounds)

    def decorate(func):
        @wraps(func)
        def bounded_(x: NDArray[Number]):
            if hasattr(func, 'constrained_shape'):
                bound_l, bound_r = min_bounds, max_bounds
            else:
                bound_l, bound_r = np.repeat(min_bounds[0], x.size), np.repeat(max_bounds[0], x.size)
            return func(x) \
                if (x > bound_l).all() and (x < bound_r).all() \
                else np.inf

        setattr(bounded_, 'constrained_values', True)
        setattr(bounded_, 'min_bounds', min_bounds)
        setattr(bounded_, 'max_bounds', max_bounds)

        return shape(func.shape)(bounded_) if hasattr(func, 'constrained_shape') else bounded_

    return decorate


def shape(shape_: tuple[int]):
    assert isinstance(shape_, tuple) and len(shape_) > 0

    def decorate(func):
        setattr(func, 'constrained_shape', True)
        setattr(func, 'shape', shape_)
        return func

    return decorate
