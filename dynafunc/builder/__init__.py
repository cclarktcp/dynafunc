
__all__ = [
    "NotPresent",
    "is_present",
    "FunctionUtil",
    "default_executor",
    "default_strategy",
    "compose_function",
    "compose_functions",
    "FunctionEditor",
    "PosOnlyType",
    "ArgsType",
    "VarArgsType",
    "KwOnlyType",
    "VarKwType",
    "FunctionBuilder",
]


from typing import List, Union, Callable, Any

from dynafunc.builder.util import (
    inspect_params_to_argtypes,
    FunctionUtil)

from dynafunc.compose import (
    default_executor,
    default_strategy,
    compose_function,
    compose_functions,
    Strategy,
    SerialStrategy)

from dynafunc.param import (
    FunctionParameter,
    PositionalOnly,
    Argument,
    VarArgs,
    KeywordOnly,
    VarKeywords)

from dynafunc.common import NotPresent, is_present

from dynafunc.builder.editor import (
    FunctionEditor,
    PosOnlyType,
    ArgsType,
    VarArgsType,
    KwOnlyType,
    VarKwType)

class FunctionBuilder:

    def __init__(
        self,
        handlers: Union[Callable, List[Callable]] = None,
        executor: Callable = None,
        strategy: Callable = None,
        name: str = NotPresent,
        posonly: PosOnlyType = None,
        args: ArgsType = None,
        varargs: VarArgsType = NotPresent,
        kwonly: KwOnlyType = None,
        varkw: VarKwType = NotPresent,
        globals_overwrite: dict = None,
        globals_updates: dict = None):
        
        self.signature = FunctionEditor.from_args(
            name=name,
            posonly=posonly,
            args=args,
            varargs=varargs,
            kwonly=kwonly,
            varkw=varkw,
            globals_overwrite=globals_overwrite,
            globals_updates=globals_updates,
        )
        self.handlers = handlers or list()
        self.executor = executor or default_executor
        self.strategy = strategy or default_strategy
    

    @property
    def name(self) -> str:
        return self.signature.name
    
    @name.setter
    def name(self, value: str):
        self.signature.set_name(value)


    def set_name(self, name: str):
        self.signature.set_name(name)

    def add_posonly(self, name: str, default: Any = NotPresent):
        self.signature.add_posonly(name, default)


    def add_args(self, name: str, default: Any = NotPresent):
        self.signature.add_args(name, default)


    def add_varargs(self, name: str):
        self.signature.add_varargs(name)


    def add_kwonly(self, name: str, default: Any):
        self.signature.add_kwonly(name, default)


    def add_varkw(self, name: str):
        self.signature.add_varkw(name)
    
    def add_params(
        self,
        params: Union[FunctionParameter, List[FunctionParameter]]):

        if not isinstance(params, list):
            params = [params]
        
        if any(not isinstance(param, FunctionParameter) for param in params):
            msg = "params must be an instance of FunctionParameter"
            raise TypeError(msg)
        
        for param in params:
            if isinstance(param, PositionalOnly):
                self.add_posonly(param.name, param.default)
            elif isinstance(param, Argument):
                self.add_args(param.name, param.default)
            elif isinstance(param, VarArgs):
                self.add_varargs(param.name)
            elif isinstance(param, KeywordOnly):
                self.add_kwonly(param.name, param.default)
            elif isinstance(param, VarKeywords):
                self.add_varkw(param.name)
                



    def reset_signature(
        self,
        posonly: bool = True,
        args: bool = True,
        varargs: bool = True,
        kwonly: bool = True,
        varkw: bool = True,
    ):
        self.signature.reset_signature(
            posonly=posonly,
            args=args,
            varargs=varargs,
            kwonly=kwonly,
            varkw=varkw,
        )

    def set_signature(
        self,
        posonly: PosOnlyType = NotPresent,
        args: ArgsType = NotPresent,
        varargs: VarArgsType = NotPresent,
        kwonly: KwOnlyType = NotPresent,
        varkw: VarKwType = NotPresent,
    ):
        kw = dict()
        if is_present(posonly):
            kw["posonly"] = posonly
        if is_present(args):
            kw["args"] = args
        if is_present(varargs):
            kw["varargs"] = varargs
        if is_present(kwonly):
            kw["kwonly"] = kwonly
        if is_present(varkw):
            kw["varkw"] = varkw
        
        self.signature.set_signature(**kw)
    
    def update_signature(
        self,
        posonly: PosOnlyType = NotPresent,
        args: ArgsType = NotPresent,
        varargs: VarArgsType = NotPresent,
        kwonly: KwOnlyType = NotPresent,
        varkw: VarKwType = NotPresent,
    ):
        kw = dict()
        if is_present(posonly):
            kw["posonly"] = posonly
        if is_present(args):
            kw["args"] = args
        if is_present(varargs):
            kw["varargs"] = varargs
        if is_present(kwonly):
            kw["kwonly"] = kwonly
        if is_present(varkw):
            kw["varkw"] = varkw
        
        self.signature.update_signature(**kw)



    def update_signature_from_params(self, params):
        return self._sig_from_params(self.update_signature, params)

    def set_signature_from_params(self, params):
        return self._sig_from_params(self.set_signature, params)
    

    def update_signature_from_function(self, function: Callable):
        return self._sig_from_function(self.update_signature, function)
        
    def set_signature_from_function(self, function: Callable):
        return self._sig_from_function(self.set_signature, function)


    def _sig_from_params(self, updater, params):
        args = inspect_params_to_argtypes(params)
        args_ = dict()
        for k,v in args.items():
            if not v:
                args_[k] = NotPresent
            else:
                args_[k] = v
        
        return updater(**args_)
        
    def _sig_from_function(self, updater, func):
        funcutil = FunctionUtil(func)
        params = funcutil.inspect_params
        return self._sig_from_params(updater, params)


    def add_handler(self, handler: Callable):
        self.handlers.append(handler)
    

    def update_globals(self, key: str, value: Any):
        return self.signature.update_globals(key, value)
    

    def overwrite_globals(self, global_vars: dict):
        return self.signature.overwrite_globals(global_vars)


    def make_signature(self) -> Callable:
        sigfunc = self.signature.apply()
        return sigfunc

    def compose(self) -> Callable:
        if not self.handlers:
            raise ValueError("there must be at least one handler present")
        
        sigfunc = self.make_signature()

        if len(self.handlers) == 1:
            return compose_function(sigfunc, self.handlers[0], self.executor)
        else:
            return compose_functions(sigfunc, self.handlers, self.executor,
                                     self.strategy, self.silent)

