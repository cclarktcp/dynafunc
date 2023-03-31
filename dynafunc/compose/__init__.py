"""compose dynamic functions by combining a signature with one or more handlers

"""

__all__ = [
    "default_executor",
    "Strategy",
    "SerialStrategy",
    "default_strategy",
    "compose_function",
    "compose_functions",
]

import inspect
from functools import wraps
from typing import List, Callable

from dynafunc.compose.executor import default_executor
from dynafunc.compose.strategy import (
    Strategy,
    SerialStrategy,
    default_strategy)
from dynafunc.common import NotPresent, is_present


COMPOSE_WRAPS_ASSIGN = (
    '__module__', '__name__', '__qualname__', '__doc__',
    '__annotations__', '__defaults__')


def compose_function(
    sig: Callable,
    handler: Callable,
    executor: Callable = default_executor):
    """compose a function with a single handler

    Args:
        sig: function signature
        handler: handler for the function
        executor: executor that carries out function execution

    Returns:
        wrapper for `sig` function's signature whose execution is carried
        out by `handler`

    """
    funcsig = inspect.signature(sig)
    @wraps(sig, assigned=COMPOSE_WRAPS_ASSIGN)
    def wrapper(*args, **kwargs):
        binding = funcsig.bind_partial(*args, **kwargs)
        binding.apply_defaults()
        args = binding.args 
        kwargs = binding.kwargs
        print(f"compose_function wrapper args={args} kwargs={kwargs}")
        return executor(sig, handler, args, kwargs)
    return wrapper


def compose_functions(
    sig: Callable,
    handlers: List[Callable],
    executor: Callable = default_executor,
    strategy: Callable = default_strategy):
    """compose a function with multiple handlers

    Args:
        sig: function signature
        handlers: handlers for the function
        executor: executor that carries out function execution
        strategy: function that carries out running of the multiple handlers

    Returns:
        wrapper for `sig` function's signature whose execution is carried
        out by `handlers`. This wrapper will return a list of tuples in the 
        form (handler, result). If silent is True, the `result` can be an
        exception

    Raises:
        if silent is False and a handler raises an exception, that exception
        is raised

    """
    funcsig = inspect.signature(sig)
    @wraps(sig, assigned=COMPOSE_WRAPS_ASSIGN)
    def wrapper(*args, **kwargs):
        binding = funcsig.bind_partial(*args, **kwargs)
        binding.apply_defaults()
        args = binding.args 
        kwargs = binding.kwargs
        return strategy(sig, handlers, executor, args, kwargs)

    return wrapper