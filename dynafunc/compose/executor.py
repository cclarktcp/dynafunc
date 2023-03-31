"""executors of function handlers

An executor takes input to a function and runs the function's handler with
the args passed to the function. Executors should have the following signature:
    func(sig: Callable, handler: Callable, args: tuple, kwargs: dict) -> Any

View the default_executor for a reference implementation
"""
import inspect
from typing import Callable


def default_executor(
    sig: Callable,
    handler: Callable,
    args: tuple = None,
    kwargs: dict = None):
    """default executor for function handlers

    Args:
        sig: function signature
        handler: function handler to execute
        args: positional args to the function
        kwargs: keyword args to the function
    
    Returns:
        result of handler(*args, **kwargs)

    Raises:
        TypeError: if given *args, **kwargs do not align with sig's parameters
        Will also raise anything raised by the execution of handler

    """
    args = args or tuple()
    kwargs = kwargs or dict()
    try:
        inspect.getcallargs(sig, *args, **kwargs)
        return handler(*args, **kwargs)
    except Exception as e:
        raise e