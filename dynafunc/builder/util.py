import inspect
import operator
import functools
import collections
from types import FunctionType
from dynafunc.common import NotPresent, is_present

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


def make_param(type, name, default=NotPresent, annotation=NotPresent):
    param_type = get_inspect_param_type(type)
    kw = dict()
    if is_present(default):
        kw['default'] = default
    if is_present(annotation):
        kw['annotation'] = annotation

    return inspect.Parameter(name, param_type, **kw)


def _make_params(args=None, varargs=None, kwonly=None, 
                 varkw=None, posonly=None):
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
    buckets = collections.OrderedDict(posonly=list(), args=list(),
                                      varargs=list(), kwonly=list(),
                                      varkw=list())
    typemap = collections.OrderedDict(posonly=list(), args=list(),
                                      varargs=NotPresent, kwonly=dict(),
                                      varkw=NotPresent)
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
        buckets = collections.OrderedDict(posonly=list(), args=list(),
                                          varargs=list(), kwonly=list(),
                                          varkw=list())
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


def make_inspect_params(func, args=None, varargs=None, kwonly=None,
                        varkw=None, posonly=None):
    funcutil = _get_funcutil(func)
    sig = funcutil.signature
    params = dict(sig.parameters)
    spec = funcutil.argspec

    extra_params = _make_params(args=args, varargs=varargs, kwonly=kwonly,
                                varkw=varkw, posonly=posonly)

    if varargs and spec.varargs:
        del params[spec.varargs]
    if varkw and spec.varkw:
        del params[spec.varkw]

    params.update(extra_params)
    return params


def update_sig(func, args=None, varargs=None, kwonly=None, varkw=None,
               posonly=None):
    params = make_inspect_params(func=func, args=args, varargs=varargs,
                                 kwonly=kwonly,varkw=varkw, posonly=posonly)
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
    bitmap = functools.reduce(_co_flags_reducer, bitmaps, dict())
    return bitmap_to_code_flags(bitmap)


def _make_code_props_for_funcsig(func, args=None, varargs=None, kwonly=None,
                                 varkw=None, posonly=None,
                                 func_name=NotPresent):
    """
    """
    funcutil = _get_funcutil(func)
    new_sig = update_sig(func=funcutil, args=args, varargs=varargs,
                         kwonly=kwonly, varkw=varkw, posonly=posonly)
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


def _make_code_props_for_funcbody(func, args=None, varargs=None, kwonly=None,
                                  varkw=None, posonly=None,
                                  func_name=NotPresent):
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
    new_sig = update_sig(func=funcutil, args=args, varargs=varargs,
                         kwonly=kwonly, varkw=varkw, posonly=posonly)
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


def _make_code_props(func, args=None, varargs=None, kwonly=None, varkw=None,
                     posonly=None, func_name=NotPresent):
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
    new_sig = update_sig(func=funcutil, args=args, varargs=varargs,
                         kwonly=kwonly, varkw=varkw, posonly=posonly)
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


def make_code(func,  args=None, varargs=None, kwonly=None, varkw=None,
              posonly=None, func_name=None, keep_sig=True, keep_body=True,
              return_props=False):
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

    code_props = code_prop_generator(func=funcutil, args=args, varargs=varargs,
                                     kwonly=kwonly, varkw=varkw,
                                     posonly=posonly, func_name=func_name)

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


def splice_funcs(sig, body, args=None, varargs=None, kwonly=None, varkw=None,
                 posonly=None, func_name=None, return_props=False):
    
    sig_code_props = make_code(sig, args=args, varargs=varargs, kwonly=kwonly,
                               varkw=varkw, posonly=posonly,
                               func_name=func_name, keep_sig=True, 
                               keep_body=False, return_props=True)
    
    body_code_props = make_code(body, args=args, varargs=varargs,
                                kwonly=kwonly, varkw=varkw, posonly=posonly,
                                func_name=func_name, keep_sig=False,
                                keep_body=True, return_props=True)
    
    code_props = combine_code_props(sig_code_props, body_code_props)

    if return_props:
        return code_props

    bodutil = _get_funcutil(body)
    return bodutil.copy_code(**code_props)



def make_function(code, global_vars=None, name=NotPresent,
                  argdefs=None, closure=None) -> FunctionType:
    """construct a new function from its parts
    """
    if global_vars is None:
        global_vars = globals()

    if not is_present(name):
        name = None
    return FunctionType(code, global_vars, name, argdefs, closure)


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
