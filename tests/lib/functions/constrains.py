from numbers import Number

from numpy.typing import NDArray

# TODO: consider introducing real constrain checks before decorated function execution


def bounds(min_bounds: NDArray[Number] | None = None, max_bounds: NDArray[Number] | None = None):
    if min_bounds and max_bounds:
        assert all(min_bounds <= max_bounds)

    def decorate(func):
        setattr(func, 'constrained_values', True)
        setattr(func, 'min_bounds', min_bounds)
        setattr(func, 'max_bounds', max_bounds)
        return func

    return decorate


def shape(shape_: tuple[int]):
    assert isinstance(shape_, tuple) and len(shape_) > 0

    def decorate(func):
        setattr(func, 'constrained_shape', True)
        setattr(func, 'shape', shape_)
        return func

    return decorate
