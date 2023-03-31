"""orchestrate execution of multiple handler executors

A strategy is the way multiple function handlers run. The default_strategy is
a basic serial execution of each 
"""
from typing import List, Callable


class Strategy:

    def __call__(
        self,
        sig: Callable,
        handlers: List[Callable],
        executor: Callable,
        args: tuple = None,
        kwargs: dict = None):
        
        raise NotImplemented()


class SerialStrategy(Strategy):

    def __init__(self, silent: bool = False):
        self.silent = silent

    def __call__(
        self,
        sig: Callable,
        handlers: List[Callable],
        executor: Callable,
        args: tuple = None,
        kwargs: dict = None):
        """serial strategy for executing multi-handler functions

        Args:
            sig: function signature
            handlers: handlers for the function
            executor: executor that carries out function execution
            args: function positional args
            kwargs: function keyword args

        Returns:
            list of tuples in the form (handler, result). If silent is
            True, the `result` can be an exception

        Raises:
            if self.silent is False and a handler raises an exception, that
            exception is raised

        """
        args = args or tuple()
        kwargs = kwargs or dict()
        results = list()
        for handler in handlers:
            try:
                result = executor(sig, handler, args, kwargs)
                results.append((handler, result))
            except Exception as e:
                if self.silent:
                    results.append((handler, e))
                else:
                    raise e
        return results



default_strategy = SerialStrategy(silent=False)