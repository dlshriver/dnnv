from __future__ import annotations

import types
from typing import Callable

import numpy as np


def get_callable_name(f: Callable) -> str:
    if isinstance(f, types.LambdaType) and (
        f.__name__ == "<lambda>" or f.__qualname__ == "<lambda>"
    ):
        unique_name = f"<lambda id={hex(id(f))}>"
        return f"{f.__module__}.{unique_name}"
    elif isinstance(f, types.BuiltinFunctionType):
        return f"{f.__qualname__}"
    elif isinstance(f, types.FunctionType):
        return f"{f.__module__}.{f.__qualname__}"
    elif isinstance(f, types.MethodType):
        return f"{f.__self__.__module__}.{f.__qualname__}"
    elif isinstance(f, np.ufunc):
        return f"numpy.{f.__name__}"
    elif isinstance(f, type) and callable(f):
        return f"{f.__module__}.{f.__qualname__}"
    else:
        raise ValueError(f"Unsupported callable type: {type(f)}")


__all__ = ["get_callable_name"]
