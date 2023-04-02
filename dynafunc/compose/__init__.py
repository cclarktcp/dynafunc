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


def _function_defaults(signature, only_present: bool = True):
    defs = dict()
    for k,v in signature.parameters.items():
        default = v.default
        if default is inspect._empty and not only_present:
            defs[k] = NotPresent
        elif default is not inspect._empty:
            defs[k] = default
    return defs 


def _real_args_kwargs(signature, *args, **kwargs):
    binding = signature.bind_partial(*args, **kwargs)
    given_args = binding.arguments
    default_args = _function_defaults(signature, only_present=True)
    need_to_pass = set(default_args) - set(given_args)
    kw = {k:default_args[k] for k in need_to_pass}
    kw.update(kwargs)
    return args, kw



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
        args_, kwargs_ = _real_args_kwargs(funcsig, *args, **kwargs)
        return executor(sig, handler, args_, kwargs_)
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

    """
    funcsig = inspect.signature(sig)
    @wraps(sig, assigned=COMPOSE_WRAPS_ASSIGN)
    def wrapper(*args, **kwargs):
        args_, kwargs_ = _real_args_kwargs(funcsig, *args, **kwargs)
        return strategy(sig, handlers, executor, args_, kwargs_)

    return wrapper