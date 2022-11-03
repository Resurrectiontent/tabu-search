from functools import wraps


def return_self_method(func):
    assert callable(func), f'Argument func should be callable. Type {type(func).__name__} is not callable.'

    @wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        return func.__self__
    return wrapper
