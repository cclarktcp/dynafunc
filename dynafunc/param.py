"""Function parameters

Module: dynafunc.param
File: dynafunc/param.py
"""
from typing import Any
from inspect import Parameter
from dataclasses import dataclass

from dynafunc.common import NotPresent, is_present



@dataclass
class FunctionParameter:
    """generic function parameter
    """


@dataclass
class PositionalOnly(FunctionParameter):
    """positional only function parameter

    func(posonly, /, ...)
    """
    name: str
    default: Any = NotPresent

    def __repr__(self):
        argstr = self.name
        if self.default is not NotPresent:
            argstr += f"={self.default}"
        return f"<{type(self).__name__} ({argstr}, /)>"


@dataclass
class Argument(FunctionParameter):
    """function parameter that can be specified positonally or by keyword

    func(arg0, arg1)
    """
    name: str
    default: Any = NotPresent

    def __repr__(self):
        argstr = self.name
        if self.default is not NotPresent:
            argstr += f"={self.default}"
        return f"<{type(self).__name__} ({argstr})>"



@dataclass
class KeywordOnly(FunctionParameter):
    """function parameter that can only be specified by keyword

    func(arg, *, kwonly=<default>)
    """
    name: str
    default: Any

    def __repr__(self):
        argstr = f"{self.name}={self.default}"
        return f"<{type(self).__name__} (*, {argstr})>"



@dataclass
class VarArgs(FunctionParameter):
    """*args

    func(*args)
    """
    name: str
    

    def __repr__(self):
        argstr = f"*{self.name}"
        return f"<{type(self).__name__} ({argstr})>"


@dataclass
class VarKeywords(FunctionParameter):
    """**kwargs

    func(**kwargs)
    """
    name: str

    def __repr__(self):
        argstr = f"**{self.name}"
        return f"<{type(self).__name__} ({argstr})>"


INSPECT_PARAM_KIND_TO_CLS = {
    Parameter.POSITIONAL_ONLY: PositionalOnly,
    Parameter.POSITIONAL_OR_KEYWORD: Argument,
    Parameter.KEYWORD_ONLY: KeywordOnly,
    Parameter.VAR_KEYWORD: VarKeywords,
    Parameter.VAR_POSITIONAL: VarArgs,
}

def inspect_param_cls(param: Parameter):
    return INSPECT_PARAM_KIND_TO_CLS.get(param.kind)

def from_inspect_param(param: Parameter):
    paramcls = inspect_param_cls(param)

    name = param.name
    default = param.default
