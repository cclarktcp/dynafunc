import copy
import inspect
import operator
import functools
import collections

from types import FunctionType
from typing import List, Dict, Tuple, Union, Callable

from dynafunc.compose import (
    default_executor,
    default_strategy,
    compose_function,
    compose_functions,
)

class NotPresent:
    pass


def is_present(val):
    return val is not NotPresent


INSPECT_PARAM_TYPE_MAP = {
    "kwonly": inspect.Parameter.KEYWORD_ONLY,
    "posonly": inspect.Parameter.POSITIONAL_ONLY,
    "args": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "varkw": inspect.Parameter.VAR_KEYWORD,
    "varargs": inspect.Parameter.VAR_POSITIONAL,
}

CODE_FLAGS_MAP = {
    "CO_OPTIMIZED": 1,
    "CO_NEWLOCALS": 2,
    "CO_VARARGS": 4,
    "CO_VARKEYWORDS": 8,
    "CO_NESTED": 16,
    "CO_GENERATOR": 32,
    "CO_NOFREE": 64,
    "CO_COROUTINE": 128,
    "CO_ITERABLE_COROUTINE": 256,
    "CO_ASYNC_GENERATOR": 512
}

CODE_FLAGS_COMBINE_OPERATOR_MAP = {
    "CO_OPTIMIZED": operator.or_,
    "CO_NEWLOCALS": operator.or_,
    "CO_VARARGS": operator.or_,
    "CO_VARKEYWORDS": operator.or_,
    "CO_NESTED": operator.or_,
    "CO_GENERATOR": operator.or_,
    "CO_NOFREE": operator.or_,
    "CO_COROUTINE": operator.or_, 
    "CO_ITERABLE_COROUTINE": operator.or_,
    "CO_ASYNC_GENERATOR": operator.or_,
}

def code_flags_to_bitmap(co_flags):
    bitmap = dict()
    for flag, num in CODE_FLAGS_MAP.items():
        bitmap[flag] = bool(co_flags & num)
    return bitmap

def bitmap_to_code_flags(bitmap):
    co_flags = 0
    for k,v in bitmap.items():
        if bool(v):
            num = CODE_FLAGS_MAP.get(k)
        else:
            num = 0
        co_flags |= num
    return co_flags



    
def get_inspect_param_type(type):
    if type not in INSPECT_PARAM_TYPE_MAP:
        typestr = ', '.join(INSPECT_PARAM_TYPE_MAP)
        msg = f"invalid type '{type}', valid types are {typestr}"
        raise KeyError(msg)
    return INSPECT_PARAM_TYPE_MAP.get(type)


def inspect_params_type_to_name(paramtype):
    _rev = {v:k for k,v in INSPECT_PARAM_TYPE_MAP.items()}
    if paramtype not in _rev:
        typestr = ', '.join([str(k) for k in _rev])
        msg = f"invalid param type '{paramtype}', valid types are {typestr}"
        raise KeyError(msg)
    return _rev.get(paramtype)


def _get_funcutil(func_or_funcutil):
    if isinstance(func_or_funcutil, FunctionUtil):
        return func_or_funcutil
    return FunctionUtil(func_or_funcutil)



def make_param(
    type,
    name,
    default=NotPresent,
    annotation=NotPresent,
):
    param_type = get_inspect_param_type(type)
    kw = dict()
    if is_present(default):
        kw['default'] = default
    if is_present(annotation):
        kw['annotation'] = annotation

    return inspect.Parameter(name, param_type, **kw)


def _make_params(
    args=None,
    varargs=None,
    kwonly=None,
    varkw=None,
    posonly=None,
):
    params = dict()
    if args:
        for arg in args:
            if isinstance(arg, tuple):
                name, default = arg
            else:
                name, default = arg, NotPresent
            p = make_param('args', name, default)
            params[name] = p 
    
    if varargs and is_present(varargs):
        print(varargs)
        if isinstance(varargs, tuple):
            name, default = varargs
        else:
            name, default = varargs, NotPresent
        p = make_param('varargs', name, default)
        params[name] = p
    
    if kwonly:
        for name, default in kwonly.items():
            p = make_param('kwonly', name, default)
            params[name] = p
    
    if varkw and is_present(varargs):
        p = make_param('varkw', varkw)
        params[varkw] = p
    
    if posonly:
        for arg in posonly:
            if isinstance(arg, tuple):
                name, default = arg
            else:
                name, default = arg, NotPresent
            p = make_param('args', name, default)
            params[name] = p
    return params


def param_to_default_form(param):
    paramtype = inspect_params_type_to_name(param.kind)

    if paramtype in ['varargs', 'varkw']:
        return param.name
    elif paramtype in ['args', 'posonly', 'kwonly']:
        name = param.name
        default = param.default
        if default is inspect._empty:
            default = NotPresent
        return name, default

    raise TypeError(f"param type '{paramtype}' is invalid")

    

def inspect_params_to_argtypes(params):
    
    buckets = collections.OrderedDict(
        posonly=list(),
        args=list(),
        varargs=list(),
        kwonly=list(),
        varkw=list(),
    )
    typemap = collections.OrderedDict(
        posonly=list(),
        args=list(),
        varargs=NotPresent,
        kwonly=dict(),
        varkw=NotPresent,
    )
    if isinstance(params, dict):
        params = list(params.values())
    
    for p in params:
        typename = inspect_params_type_to_name(p.kind)
        bucket = buckets.get(typename)
        bucket.append(p)
    
    for paramtype, parameters in buckets.items():
        if not parameters:
            continue

        conv = [param_to_default_form(p) for p in parameters]

        if paramtype in ['varargs', 'varkw']:
            if len(conv) == 1:
                conv = conv[0]
            typemap[paramtype] = conv
        elif paramtype in ['args', 'posonly']:
            typemap[paramtype] += conv
        elif paramtype == 'kwonly':
            conv = dict(conv)
            typemap[paramtype].update(conv)
    
    return typemap


    

def sort_inspect_params(params, buckets=None):
    if buckets is None:
        buckets = collections.OrderedDict(
            posonly=list(),
            args=list(),
            varargs=list(),
            kwonly=list(),
            varkw=list(),
        )
    if isinstance(params, dict):
        params = list(params.values())
    
    for p in params:
        typename = inspect_params_type_to_name(p.kind)
        bucket = buckets.get(typename)
        bucket.append(p)
    
    sorted = [
        p for typename, bucket in buckets.items()
        for p in bucket
    ]

    return sorted

    
def _make_co_varnames(sig, extravars=None):
    buckets = collections.OrderedDict(
        posonly=list(),
        args=list(),
        kwonly=list(),
        varargs=list(),
        varkw=list(),
    )
    params = list(sig.parameters.values())
    params = sort_inspect_params(params, buckets)

    co_varnames = [p.name for p in params]

    if extravars:
        co_varnames += extravars
    
    return tuple(co_varnames)


def defaults_from_inspect_params(params):
    if isinstance(params, dict):
        params = list(params.values())
    
    params = sort_inspect_params(params)
    
    argdefaults = list()
    kwonlydefaults = dict()

    for p in params:
        typename = inspect_params_type_to_name(p.kind)
        if typename in ['args', 'posonly']:
            default = p.default
            if default is inspect._empty:
                continue
            argdefaults.append(default)
        elif typename == "kwonly":
            key = p.name
            default = p.default
            kwonlydefaults[key] = default

    if not argdefaults:
        argdefaults = None
    else:
        argdefaults = tuple(argdefaults)

    if not kwonlydefaults:
        kwonlydefaults = None

    return argdefaults, kwonlydefaults





def make_inspect_params(
    func,
    args=None,
    varargs=None,
    kwonly=None,
    varkw=None,
    posonly=None,
):
    funcutil = _get_funcutil(func)
    sig = funcutil.signature
    params = dict(sig.parameters)
    spec = funcutil.argspec

    extra_params = _make_params(
        args=args,
        varargs=varargs,
        kwonly=kwonly,
        varkw=varkw,
        posonly=posonly,
    )

    if varargs and spec.varargs:
        del params[spec.varargs]
    if varkw and spec.varkw:
        del params[spec.varkw]

    params.update(extra_params)
    return params


def update_sig(
    func,
    args=None,
    varargs=None,
    kwonly=None,
    varkw=None,
    posonly=None,
):
    params = make_inspect_params(
        func=func,
        args=args,
        varargs=varargs,
        kwonly=kwonly,
        varkw=varkw,
        posonly=posonly,
    )
    funcutil = _get_funcutil(func)
    sig = funcutil.signature
    param_list = list(params.values())
    param_list = sort_inspect_params(param_list)
    return sig.replace(parameters=param_list)
    


def _make_code_props_argcounts(sig):
    props = {
        "co_argcount": 0,
        "co_kwonlyargcount": 0,
        "co_posonlyargcount": 0,
    }

    for name, param in sig.parameters.items():
        typename = inspect_params_type_to_name(param.kind)
        if typename == "kwonly":
            props['co_kwonlyargcount'] += 1
        if typename in ['posonly', 'args']:
            props['co_argcount'] += 1
        if typename == 'posonly':
            props['co_posonlyargcount'] += 1
    
    return props
    

def _remove_code_specific_co_flags(co_flags):
    todel = [
        "CO_NOFREE",
        "CO_NESTED",
        "CO_GENERATOR",
        "CO_COROUTINE",
        "CO_ITERABLE_COROUTINE",
        "CO_ASYNC_GENERATOR",
    ]
    bitmap = code_flags_to_bitmap(co_flags)
    for key in todel:
        if key in bitmap:
            bitmap[key] = False
    
    return bitmap_to_code_flags(bitmap)

def _remove_sig_specific_co_flags(co_flags):
    todel = [
        "CO_VARARGS",
        "CO_VARKEYWORDS",
    ]
    bitmap = code_flags_to_bitmap(co_flags)
    for key in todel:
        if key in bitmap:
            bitmap[key] = False
    
    return bitmap_to_code_flags(bitmap)
    


def _make_co_flags(func, sig):
    """
        CO_OPTIMIZED
        CO_NEWLOCALS
        CO_VARARGS
        CO_VARKEYWORDS
        CO_NESTED
        CO_GENERATOR
        CO_NOFREE
        CO_COROUTINE
        CO_ITERABLE_COROUTINE
        CO_ASYNC_GENERATOR
    """
    funcutil = _get_funcutil(func)
    code_props = funcutil.code_props
    co_flags = code_props.get("co_flags")
    bitmap = funcutil.code_flags_bitmap

    varargs = False
    varkw = False
    for name, param in sig.parameters.items():
        paramtype = inspect_params_type_to_name(param.kind)
        if paramtype == "varargs":
            varargs = True
        elif paramtype == "varkw":
            varkw = True
    
    bitmap["CO_VARARGS"] = varargs
    bitmap["CO_VARKEYWORDS"] = varkw

    return bitmap_to_code_flags(bitmap)


def _co_flags_reducer(bitmap1, bitmap2):
    bitmap = dict()
    keys = set(bitmap1).union(bitmap2)
    for k in keys:
        op = CODE_FLAGS_COMBINE_OPERATOR_MAP.get(k)
        if not op:
            msg = f"invalid co_flag '{k}' valid options are "
            msg += ', '.join(CODE_FLAGS_COMBINE_OPERATOR_MAP)
            raise KeyError(msg)
        v1 = bitmap1.get(k, False)
        v2 = bitmap2.get(k, False)
        bitmap[k] = op(v1, v2)
    
    return bitmap



def combine_co_flags(*co_flags):
    bitmaps = [
        code_flags_to_bitmap(flagset) for flagset in co_flags
    ]
    print(co_flags, bitmaps)
    bitmap = functools.reduce(_co_flags_reducer, bitmaps, dict())
    return bitmap_to_code_flags(bitmap)




def _make_code_props_for_funcsig(
    func,
    args=None,
    varargs=None,
    kwonly=None,
    varkw=None,
    posonly=None,
    func_name=NotPresent, 
):
    """
    """
    funcutil = _get_funcutil(func)
    new_sig = update_sig(
        func=funcutil,
        args=args,
        varargs=varargs,
        kwonly=kwonly,
        varkw=varkw,
        posonly=posonly,
    )
    code_props = dict()
    argcounts = _make_code_props_argcounts(new_sig)
    co_flags = _make_co_flags(funcutil, new_sig)
    co_flags = _remove_code_specific_co_flags(co_flags)
    co_varnames = _make_co_varnames(new_sig)
    co_names = funcutil.code_props.get("co_names")
    if co_names:
        co_names = tuple(n for n in co_names if n not in co_varnames)

    nlocals = len(co_varnames)

    code_props['co_nlocals'] = nlocals

    code_props.update(argcounts)
    code_props['co_flags'] = co_flags
    code_props['co_varnames'] = co_varnames
    code_props["co_names"] = co_names

    if func_name and is_present(func_name):
        code_props['co_name'] = func_name
    
    return code_props


def _make_code_props_for_funcbody(
    func,
    args=None,
    varargs=None,
    kwonly=None,
    varkw=None,
    posonly=None,
    func_name=NotPresent, 
):
    """
        {'co_argcount': 2,
        'co_cellvars': (),
        'co_code': b'd\x01}\x07|\x00|\x02|\x03f\x03S\x00',
        'co_consts': (None, 1),
        'co_filename': '<ipython-input-464-c7cf22b651d4>',
        'co_firstlineno': 1,
        'co_flags': 79,
        'co_freevars': (),
        'co_kwonlyargcount': 3,
        'co_lines': <function code.co_lines>,
        'co_linetable': b'\x04\x01\n\x01',
        'co_lnotab': b'\x00\x01\x04\x01',
        'co_name': 'blahblah2',
        'co_names': (),
        'co_nlocals': 8,
        'co_posonlyargcount': 1,
        'co_stacksize': 3,
        'co_varnames': ('name',
        'name2',
        'kw0',
        'kw1',
        'bruh',
        'args',
        'kwargs',
        'one'),

        co_argcount
        co_flags
        co_kwonlyargcount
        co_posonlyargcount

    """
    funcutil = _get_funcutil(func)
    new_sig = update_sig(
        func=funcutil,
        args=args,
        varargs=varargs,
        kwonly=kwonly,
        varkw=varkw,
        posonly=posonly,
    )
    code_props = dict()
    co_flags = _make_co_flags(funcutil, new_sig)
    co_flags = _remove_sig_specific_co_flags(co_flags)

    co_varnames = funcutil.extravars
    co_names = funcutil.code_props.get("co_names")
    if co_names:
        co_names = tuple(n for n in co_names if n not in co_varnames)

    nlocals = len(co_varnames)

    code_props['co_nlocals'] = nlocals
    code_props['co_flags'] = co_flags
    code_props['co_varnames'] = co_varnames
    code_props["co_names"] = co_names


    


    if func_name and is_present(func_name):
        code_props['co_name'] = func_name
    
    return code_props



def _make_code_props(
    func,
    args=None,
    varargs=None,
    kwonly=None,
    varkw=None,
    posonly=None,
    func_name=NotPresent, 
):
    """
        {'co_argcount': 2,
        'co_cellvars': (),
        'co_code': b'd\x01}\x07|\x00|\x02|\x03f\x03S\x00',
        'co_consts': (None, 1),
        'co_filename': '<ipython-input-464-c7cf22b651d4>',
        'co_firstlineno': 1,
        'co_flags': 79,
        'co_freevars': (),
        'co_kwonlyargcount': 3,
        'co_lines': <function code.co_lines>,
        'co_linetable': b'\x04\x01\n\x01',
        'co_lnotab': b'\x00\x01\x04\x01',
        'co_name': 'blahblah2',
        'co_names': (),
        'co_nlocals': 8,
        'co_posonlyargcount': 1,
        'co_stacksize': 3,
        'co_varnames': ('name',
        'name2',
        'kw0',
        'kw1',
        'bruh',
        'args',
        'kwargs',
        'one'),

        co_argcount
        co_flags
        co_kwonlyargcount
        co_posonlyargcount

    """
    funcutil = _get_funcutil(func)
    new_sig = update_sig(
        func=funcutil,
        args=args,
        varargs=varargs,
        kwonly=kwonly,
        varkw=varkw,
        posonly=posonly,
    )
    code_props = dict()
    argcounts = _make_code_props_argcounts(new_sig)
    co_flags = _make_co_flags(funcutil, new_sig)
    co_varnames = _make_co_varnames(new_sig, funcutil.extravars)

    old_code_props = funcutil.code_props
    old_varnames = old_code_props.get("co_varnames")

    diff = len(co_varnames) - len(old_varnames)

    nlocals = old_code_props.get("co_nlocals")
    nlocals += diff

    co_names = old_code_props.get("co_names")
    if co_names:
        co_names = tuple(n for n in co_names if n not in co_varnames)

    code_props['co_nlocals'] = nlocals

    code_props.update(argcounts)
    code_props['co_flags'] = co_flags
    code_props['co_varnames'] = co_varnames
    code_props["co_names"] = co_names

    

    if func_name and is_present(func_name):
        code_props['co_name'] = func_name
    
    return code_props


def make_code(
    func, 
    args=None,
    varargs=None,
    kwonly=None,
    varkw=None,
    posonly=None,
    func_name=None,
    keep_sig=True,
    keep_body=True,
    return_props=False,    
):
    funcutil = _get_funcutil(func)

    code_prop_generator = None
    if keep_sig and keep_body:
        code_prop_generator = _make_code_props
    elif keep_sig:
        code_prop_generator = _make_code_props_for_funcsig
    elif keep_body:
        code_prop_generator = _make_code_props_for_funcbody
    else:
        raise ValueError(
            "one of or both keep_sig and keep_body must be True"
        )

    code_props = code_prop_generator(
        func=funcutil,
        args=args,
        varargs=varargs,
        kwonly=kwonly,
        varkw=varkw,
        posonly=posonly,
        func_name=func_name,
    )

    if return_props:
        return code_props

    return funcutil.copy_code(**code_props)
    

def _code_prop_reducer(props1, props2):
    props = dict(**props2)
    props.update(props1)
    flags1 = props1.get("co_flags", 0)
    flags2 = props2.get("co_flags", 0)
    varnames1 = props1.get("co_varnames", tuple())
    varnames2 = props2.get("co_varnames", tuple())

    vars1 = list(varnames1) if varnames1 else list()
    vars2 = list(varnames2) if varnames2 else list()

    vars_ = vars1 + vars2

    co_names1 = props1.get("co_names") or tuple()
    co_names2 = props2.get("co_names") or tuple()

    
    co_names_all = list(co_names1) + list(co_names2)


    if co_names_all:
        co_names_all = tuple(n for n in co_names_all if n not in vars_)
    else:
        co_names_all = tuple()

    print(props1, props2)
    flags = combine_co_flags(flags1, flags2)
    props['co_flags'] = flags
    props['co_varnames'] = tuple(vars_)
    props['co_nlocals'] = len(vars_)
    props["co_names"] = tuple(co_names_all)

    return props 



def combine_code_props(*props):
    return functools.reduce(_code_prop_reducer, props, dict())
    

def code_to_props(code):
    return {
        k:getattr(code, k) for k in dir(code)
        if not k.startswith("__") and not callable(getattr(code, k))
    }


def splice_funcs(
    sig,
    body,
    args=None,
    varargs=None,
    kwonly=None,
    varkw=None,
    posonly=None,
    func_name=None,
    return_props=False, 
):
    sig_code_props = make_code(
        sig,
        args=args,
        varargs=varargs,
        kwonly=kwonly,
        varkw=varkw,
        posonly=posonly,
        func_name=func_name,
        keep_sig=True, 
        keep_body=False,
        return_props=True,
    )
    body_code_props = make_code(
        body,
        args=args,
        varargs=varargs,
        kwonly=kwonly,
        varkw=varkw,
        posonly=posonly,
        func_name=func_name,
        keep_sig=False, 
        keep_body=True,
        return_props=True,
    )
    code_props = combine_code_props(sig_code_props, body_code_props)

    if return_props:
        return code_props

    bodutil = _get_funcutil(body)
    return bodutil.copy_code(**code_props)



def make_function(
    code,
    global_vars=None,
    name=NotPresent,
    argdefs=None,
    closure=None) -> FunctionType:
    """construct a new function from its parts
    """
    if global_vars is None:
        global_vars = globals()

    if not is_present(name):
        name = None

    return FunctionType(
        code,
        global_vars,
        name,
        argdefs,
        closure
    )


class FunctionUtil:
    def __init__(self, func):
        self.function = func


    @functools.cached_property
    def signature(self):
        return inspect.signature(self.function)


    @functools.cached_property
    def argspec(self):
        return inspect.getfullargspec(self.function)


    @functools.cached_property
    def code(self):
        return self.function.__code__

    @functools.cached_property
    def argcount(self):
        return len(self.signature.parameters)
    
    @functools.cached_property
    def extravars(self):
        return list(self.code.co_varnames[self.argcount:])
    
    @property
    def name(self):
        code_props = self.code_props
        code_props.get("co_name")
    
    @property
    def argsdefaults(self):
        return self.argspec.defaults
    

    @property
    def kwonlydefaults(self):
        return self.argspec.kwonlydefaults

    @property
    def globalvars(self):
        return self.function.__globals__

    @property
    def code_props(self):
        code = self.code
        return code_to_props(code)

    @property
    def code_flags_bitmap(self):
        co_flags = self.code_props.get("co_flags")
        return code_flags_to_bitmap(co_flags)
    
    @property
    def inspect_params(self):
        params = self.signature.parameters
        return [p for name, p in params.items()]
    
    @property
    def params_format(self):
        params = self.inspect_params
        return inspect_params_to_argtypes(params)


    def copy_code(self, **kwargs):
        return self.code.replace(**kwargs)


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
        func=None,
        name=NotPresent,
        posonly=None,
        args=None,
        varargs=NotPresent,
        kwonly=None,
        varkw=NotPresent,
        globals_overwrite=None,
        globals_updates=None,
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
    def funcutil(self):
        return FunctionUtil(self.function)
    

    @property
    def name(self):
        if not is_present(self._name):
            return self.funcutil.name
        return self._name
    
    @name.setter
    def name(self, value):
        self.set_name(value)



    def set_name(self, name):
        self._name = name
    

    def reset_signature(
        self,
        posonly=True,
        args=True,
        varargs=True,
        kwonly=True,
        varkw=True,
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
        posonly=NotPresent,
        args=NotPresent,
        varargs=NotPresent,
        kwonly=NotPresent,
        varkw=NotPresent,
    ):
        reset_kw = {
            "posonly": is_present(posonly),
            "args": is_present(args),
            "varargs": is_present(varargs),
            "kwonly": is_present(kwonly),
            "varkw": is_present(varkw),
        }
        self.reset_signature(**reset_kw)
        self.update_signature(
            posonly=posonly,
            args=args,
            varargs=varargs,
            kwonly=kwonly,
            varkw=varkw,
        )


    def update_signature(
        self,
        posonly=NotPresent,
        args=NotPresent,
        varargs=NotPresent,
        kwonly=NotPresent,
        varkw=NotPresent,
    ):
        
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
    

    def update_signature_from_function(self, function):
        return self._sig_from_function(self.update_signature, function)
        
    def set_signature_from_function(self, function):
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


    def add_posonly(self, name, default=NotPresent):
        self._add_posonly.append((name, default))

    def add_varargs(self, name):
        self._add_varargs = name
    
    def add_varkw(self, name):
        self._add_varkw = name
    
    def add_args(self, name, default=NotPresent):
        self._add_args.append((name, default))
    
    def add_kwonly(self, name, default):
        self._add_kwonly[name] = default
    

    def update_globals(self, key, value):
        self._globals_updates[key] = value
    

    def overwrite_globals(self, global_vars):
        self._globals_overwrite = global_vars
        self._globals_updates = dict()


    def get_globals(self):
        if self._globals_overwrite:
            gvars = self._globals_overwrite
        else:
            gvars = self.funcutil.globalvars
        
        

        if self._globals_updates:
            gvars = copy.copy(gvars)
            gvars.update(self._globals_updates)
        return gvars
    

    def get_apply_args(self):
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


    def new_signature(self):
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
        

    def apply(self):
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



class FunctionBuilder:

    def __init__(
        self,
        handlers: Union[Callable, List[Callable]] = None,
        executor: Callable = None,
        strategy: Callable = None,
        name=NotPresent,
        posonly=None,
        args=None,
        varargs=NotPresent,
        kwonly=None,
        varkw=NotPresent,
        globals_overwrite=None,
        globals_updates=None,
    ):
        
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
    def name(self):
        return self.signature.name
    
    @name.setter
    def name(self, value):
        self.signature.set_name(value)



    def set_name(self, name):
        self.signature.set_name(name)

    def reset_signature(
        self,
        posonly=True,
        args=True,
        varargs=True,
        kwonly=True,
        varkw=True,
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
        posonly=NotPresent,
        args=NotPresent,
        varargs=NotPresent,
        kwonly=NotPresent,
        varkw=NotPresent,
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
        posonly=NotPresent,
        args=NotPresent,
        varargs=NotPresent,
        kwonly=NotPresent,
        varkw=NotPresent,
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
    

    def update_signature_from_function(self, function):
        return self._sig_from_function(self.update_signature, function)
        
    def set_signature_from_function(self, function):
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


    def add_handler(self, handler):
        self.handlers.append(handler)
    

    def update_globals(self, key, value):
        return self.signature.update_globals(key, value)
    

    def overwrite_globals(self, global_vars):
        return self.signature.overwrite_globals(global_vars)


    def make_signature(self):
        sigfunc = self.signature.apply()
        return sigfunc

    def compose(self):
        if not self.handlers:
            raise ValueError("there must be at least one handler present")
        
        sigfunc = self.make_signature()

        if len(self.handlers) == 1:
            return compose_function(sigfunc, self.handlers[0], self.executor)
        else:
            return compose_functions(sigfunc, self.handlers, self.executor,
                                     self.strategy, self.silent)





def extend_function(
    func,
    wrapper=None,
    args=None,
    varargs=None,
    kwonly=None,
    varkw=None,
    posonly=None,
):

    FunctionEditor()
