import copy
from dynafunc.builder.util import (
    inspect_params_to_argtypes,
    update_sig,
    defaults_from_inspect_params,
    make_code,
    make_function,
    FunctionUtil)
from typing import Tuple, Any, Union, Dict, List, Callable
from dynafunc.common import NotPresent, is_present

_PosOnlyType = Union[str, Tuple[str, Any]]
PosOnlyType = Union[_PosOnlyType, List[_PosOnlyType]]
ArgsType = Union[str, Tuple[str, Any]]
VarArgsType = str
KwOnlyType = Tuple[str, Any], List[Tuple[str, Any]], Dict[str, Any]
VarKwType = str


class FunctionEditor:

    def __init__(self, func=None):
        self.function = func
        if self.function is None:
            def _default_function_(): pass
            self.function = _default_function_

        self._name = NotPresent
        self._add_args = list()
        self._add_varargs = NotPresent
        self._add_kwonly = dict()
        self._add_varkw = NotPresent
        self._add_posonly = list()

        self._globals_overwrite = dict()
        self._globals_updates = dict()
    

    @classmethod
    def from_args(
        cls,
        func: Callable = None,
        name: str = NotPresent,
        posonly: PosOnlyType = None,
        args: ArgsType = None,
        varargs: VarArgsType = NotPresent,
        kwonly: KwOnlyType = None,
        varkw: VarKwType = NotPresent,
        globals_overwrite: dict = None,
        globals_updates: dict = None,
    ):
        o = cls(func)
        if posonly is None:
            posonly = list()
        if args is None:
            args = list()
        if kwonly is None:
            kwonly = dict()
        if globals_overwrite is None:
            globals_overwrite = dict()
        if globals_updates is None:
            globals_updates = dict()
        
        o._name = name
        o._add_posonly += posonly
        o._add_args += args
        o._add_varargs = varargs
        o._add_kwonly.update(kwonly)
        o._add_varkw = varkw
        o._globals_overwrite = globals_overwrite
        o._globals_updates = globals_updates

        return o
    

    @property
    def funcutil(self) -> FunctionUtil:
        return FunctionUtil(self.function)
    

    @property
    def name(self) -> str:
        if not is_present(self._name):
            return self.funcutil.name
        return self._name
    
    @name.setter
    def name(self, value: str):
        self.set_name(value)



    def set_name(self, name: str):
        self._name = name
    

    def reset_signature(
        self,
        posonly: bool = True,
        args: bool = True,
        varargs: bool = True,
        kwonly: bool = True,
        varkw: bool = True,
    ):
        if posonly:
            self._add_posonly = list()
        if args:
            self._add_args = list()
        if varargs:
            self._add_varargs = NotPresent
        if kwonly:
            self._add_kwonly = dict()
        if varkw:
            self._add_varkw = NotPresent

    def set_signature(
        self,
        posonly: PosOnlyType = NotPresent,
        args: ArgsType = NotPresent,
        varargs: VarArgsType = NotPresent,
        kwonly: KwOnlyType = NotPresent,
        varkw: VarKwType = NotPresent):
        reset_kw = {
            "posonly": is_present(posonly),
            "args": is_present(args),
            "varargs": is_present(varargs),
            "kwonly": is_present(kwonly),
            "varkw": is_present(varkw),
        }
        self.reset_signature(**reset_kw)
        self.update_signature(posonly=posonly, args=args, varargs=varargs,
                              kwonly=kwonly, varkw=varkw)


    def update_signature(
        self,
        posonly: PosOnlyType = NotPresent,
        args: ArgsType = NotPresent,
        varargs: VarArgsType = NotPresent,
        kwonly: KwOnlyType = NotPresent,
        varkw: VarKwType = NotPresent,
    ):
        """
        PosOnlyType = Union[str, Tuple[str, Any]]
        # PosOnlyType = Union[_PosOnlyType, List[_PosOnlyType]]
        ArgsType = Union[str, Tuple[str, Any]]
        VarArgsType = str
        KwOnlyType = Tuple[str, Any], List[Tuple[str, Any]], Dict[str, Any]
        VarKwType = str

        """
        
        if is_present(posonly):
            if isinstance(posonly, tuple):
                self.add_posonly(*posonly)
            elif isinstance(posonly, str):
                self.add_posonly(posonly)
            elif isinstance(posonly, list):
                for item in posonly:
                    if isinstance(item, tuple):
                        self.add_posonly(*item)
                    elif isinstance(item, str):
                        self.add_posonly(item)
                    else:
                        raise TypeError(
                            f"invalid type for posonly: '{type(item)}'"
                        )
            else:
                raise TypeError(f"invalid type for posonly: '{type(posonly)}'")
                
            self._add_posonly += posonly

        if is_present(args):
            if isinstance(args, tuple):
                self.add_args(*args)
            elif isinstance(args, str):
                self.add_args(args)
            elif isinstance(args, list):
                for item in args:
                    if isinstance(item, tuple):
                        self.add_args(*item)
                    elif isinstance(item, str):
                        self.add_args(item)
                    else:
                        raise TypeError(
                            f"invalid type for args: '{type(item)}'"
                        )
            else:
                raise TypeError(f"invalid type for args: '{type(args)}'")


        if varargs and is_present(varargs):
            self.add_varargs(varargs)

        if is_present(kwonly):
            if isinstance(kwonly, tuple):
                self.add_kwonly(*kwonly)
            elif isinstance(kwonly, list):
                for item in kwonly:
                    if isinstance(item, tuple):
                        self.add_kwonly(*item)
                    else:
                        raise TypeError(
                            f"invalid type for kwonly: '{type(item)}'"
                        )
            elif isinstance(kwonly, dict):
                for name, default in kwonly.items():
                    self.add_kwonly(name, default)
            else:
                raise TypeError(
                    f"invalid type for kwonly: '{type(kwonly)}'"
                )

        if varkw and is_present(varkw):
            self.add_varargs(varkw)


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


    def add_posonly(self, name: str, default: Any = NotPresent):
        self._add_posonly.append((name, default))

    def add_varargs(self, name: str):
        self._add_varargs = name
    
    def add_varkw(self, name: str):
        self._add_varkw = name
    
    def add_args(self, name: str, default: Any = NotPresent):
        self._add_args.append((name, default))
    
    def add_kwonly(self, name: str, default: Any):
        self._add_kwonly[name] = default
    

    def update_globals(self, key: str, value: Any):
        self._globals_updates[key] = value
    

    def overwrite_globals(self, global_vars: dict):
        self._globals_overwrite = global_vars
        self._globals_updates = dict()


    def get_globals(self) -> dict:
        if self._globals_overwrite:
            gvars = self._globals_overwrite
        else:
            gvars = self.funcutil.globalvars
        
        

        if self._globals_updates:
            gvars = copy.copy(gvars)
            gvars.update(self._globals_updates)
        return gvars
    

    def get_apply_args(self) -> dict:
        kw = dict()

        if self._add_args and is_present(self._add_args):
            kw['args'] = self._add_args
        if self._add_varargs and is_present(self._add_varargs):
            kw['varargs'] = self._add_varargs
        if self._add_kwonly and is_present(self._add_kwonly):
            kw['kwonly'] = self._add_kwonly
        if self._add_varkw and is_present(self._add_varkw):
            kw['varkw'] = self._add_varkw
        if self._add_posonly and is_present(self._add_posonly):
            kw['posonly'] = self._add_posonly
        if self._name and is_present(self._name):
            kw['func_name'] = self._name
        
        return kw


    def new_signature(self) -> Callable:
        kw = self.get_apply_args()
        if 'func_name' in kw:
            del kw['func_name']
        new_sig = update_sig(
            func=self.funcutil,
            **kw
        )
        return new_sig

    def get_defaults(self):
        sig = self.new_signature()
        params = list(sig.parameters.values())
        return defaults_from_inspect_params(params)
        

    def apply(self) -> Callable:
        kw = self.get_apply_args()
        new_code = make_code(
            self.funcutil, 
            **kw
        )

        argdefs, kwonlydefs = self.get_defaults()
        global_vars = self.get_globals()

        func = make_function(
            new_code,
            global_vars=global_vars,
            name=self._name,
            argdefs=argdefs,
        )

        func.__kwdefaults__ = kwonlydefs
        return func 
